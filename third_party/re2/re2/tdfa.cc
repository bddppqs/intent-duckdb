// Copyright 2006-2007 The RE2 Authors (adapted from nfa.cc).
// Modifications Copyright 2026 The intent-duckdb Authors.
// Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

// Prog::SearchTDFA: anchored submatch search with a tagged DFA built lazily
// from the program, as the DFA is.
//
// A state is the ordered thread list the NFA (nfa.cc) holds at a text
// position: the ByteRange and Match instructions its epsilon closures reach,
// in priority order, each with the capture registers that hold its submatch
// slots. A transition on a byte class steps every thread whose ByteRange
// matches the class, in order, through the epsilon closure of its out(),
// exactly as NFA::AddToThreadq does (an instruction reached again in the same
// step is not visited again; the first, higher-priority arrival keeps it), and
// records the register operations that turn the source registers into the
// target's: a copy, or the position after the byte for a capture taken in the
// closure. Registers are renumbered in order of first use, so equal thread
// lists with equal register sharing are one state and the state set is
// finite; a transition that renumbers nothing changes no register, and a
// state that loops on a byte that way is scanned over without stepping.
//
// The scan of such a run takes sixteen bytes at a time (SSE2, or NEON on
// AArch64) when the state continues it on every ASCII byte but at most three
// (RE2::Options::set_tdfa_vector_scan): a block is compared with those three
// bytes, and every byte >= 0x80 ends the vector step too, so the step consumes
// only bytes the byte-by-byte scan would, which then takes over at the first
// byte that may end the run. Whether a state is admitted is read off its
// transitions, which the first scan of a run on it builds for every byte class
// with an ASCII byte.
//
// The match rules are the NFA's. When the match must end at the end of the
// text (kFullMatch, or a program anchored at the end), the first Match thread
// of the state at the end wins. Otherwise (leftmost-first), the first Match
// thread of a state wins over every thread after it, which is cut, and a later
// match of a thread before it replaces it.
//
// Programs with empty-width assertions (other than the start and end anchors,
// which the program holds as flags) are not admitted, and neither is a search
// that would grow the automata past their budget: the caller then runs the
// other engines, so the results are the same either way. The budget is a
// quarter of the program's DFA memory, at most 1MiB, taken from the DFA's
// (Prog::EnableTDFA): one per compiled pattern, shared by its automata (one
// per match rule and submatch count asked for); none is per thread.
//
// Concurrency: one search at a time runs a program's automata, under a flag
// taken with an atomic exchange (acquire) and released with a store
// (release); states and transitions are built only by that search. A search
// that finds the flag taken runs the other engines. A thread-private program
// (RE2::Options::set_tdfa_thread_private) is searched by one thread at a time
// by its owner's contract, and its searches take no flag.

#include <stdint.h>
#include <string.h>
#include <algorithm>
#include <atomic>
#include <new>
#include <string>
#include <unordered_map>
#include <vector>
#if defined(__GNUC__) && defined(__SSE2__)
#include <emmintrin.h>
#elif defined(__GNUC__) && defined(__aarch64__) && !defined(__AARCH64EB__)
#include <arm_neon.h>
#endif

#include "util/logging.h"
#include "re2/prog.h"
#include "re2/sparse_set.h"

namespace duckdb_re2 {

namespace {

// Submatch slots 2.. (groups 1..9) a search may ask for; slots 0 and 1 are
// the start of the text and the match end.
const int kMaxTrack = 2*(Prog::kMaxTDFACapture-1);
// A thread's slot in a state: a register, or unset. While a transition is
// built: also the position after the consumed byte.
const int8_t kUnset = -1;
const int8_t kNew = -2;
// Registers of one state; a state that would need more is past the budget.
const int kMaxRegs = 64;
// States of one automaton.
const int kMaxStates = 4096;
// The budget's ceiling (Prog::EnableTDFA).
const int64_t kMaxBudget = 1 << 20;
// Whether the vector step of a run's scan has a 16-byte kernel here (SSE2, or
// NEON on little-endian AArch64); elsewhere every run is scanned byte by byte.
#if defined(__GNUC__) && (defined(__SSE2__) || \
                          (defined(__aarch64__) && !defined(__AARCH64EB__)))
const bool kVectorScan = true;
#else
const bool kVectorScan = false;
#endif
// The ASCII bytes that may end a run the vector step takes.
const int kMaxStops = 3;
// TState::scan of a state whose runs are scanned byte by byte.
const uint8_t kNoScan = 3;

struct TState;

// A transition that changes registers: the target, and for each of its
// registers the source register or kNew.
struct TEdge {
  TState* to;
  int nregs;
  int8_t src[kMaxRegs];
};

struct TState {
  std::vector<int> insts;      // thread instructions, priority order
  std::vector<int8_t> slots;   // insts.size() * ntrack registers or kUnset
  int nregs = 0;
  int match = -1;              // first Match thread, or -1
  bool dead = false;           // no threads
  // bytes on which the state steps to itself: 1 changing no register, 2 with
  // self_edge's register operations (each register kept or set to the
  // position after the byte)
  uint8_t self[256];
  // the vector step of its runs (TDFA::AdmitScan): 0 not decided yet, the
  // mode of the runs it takes (1 or 2), or kNoScan; and the ASCII bytes that
  // end such a run, the unused entries 0x80 (every byte >= 0x80 ends the step)
  uint8_t scan = 0;
  uint8_t stop[kMaxStops];
  const TEdge* self_edge = NULL;
  // per byte class, in the allocation after the state: 0 not built; a TState*
  // (no register changes); a TEdge* with the low bit set
  uintptr_t* next() { return reinterpret_cast<uintptr_t*>(this+1); }
};

TState* NewState(int nclass) {
  void* mem = ::operator new(sizeof(TState) + nclass*sizeof(uintptr_t));
  TState* s = new (mem) TState;
  memset(s->self, 0, sizeof s->self);
  memset(s->next(), 0, nclass*sizeof(uintptr_t));
  return s;
}

void DeleteState(TState* s) {
  s->~TState();
  ::operator delete(s);
}

struct Thread {
  int id;
  int8_t slots[kMaxTrack];
};

// The vector step of a run: from text position j, the position of the first
// byte that is one of the stop bytes or >= 0x80, looking at whole blocks of 16
// bytes only; past the last whole block, the position after it.
inline size_t ScanBlocks(const char* begin, size_t j, size_t n,
                         const uint8_t* stop) {
#if defined(__GNUC__) && defined(__SSE2__)
  const __m128i s0 = _mm_set1_epi8(static_cast<char>(stop[0]));
  const __m128i s1 = _mm_set1_epi8(static_cast<char>(stop[1]));
  const __m128i s2 = _mm_set1_epi8(static_cast<char>(stop[2]));
  for (; j+16 <= n; j += 16) {
    const __m128i b =
        _mm_loadu_si128(reinterpret_cast<const __m128i*>(begin+j));
    const __m128i eq = _mm_or_si128(
        _mm_or_si128(_mm_cmpeq_epi8(b, s0), _mm_cmpeq_epi8(b, s1)),
        _mm_cmpeq_epi8(b, s2));
    // the sign bit of a byte >= 0x80 is set
    const int mask = _mm_movemask_epi8(_mm_or_si128(eq, b));
    if (mask != 0)
      return j + __builtin_ctz(static_cast<unsigned>(mask));
  }
#elif defined(__GNUC__) && defined(__aarch64__) && !defined(__AARCH64EB__)
  const uint8x16_t s0 = vdupq_n_u8(stop[0]);
  const uint8x16_t s1 = vdupq_n_u8(stop[1]);
  const uint8x16_t s2 = vdupq_n_u8(stop[2]);
  const uint8x16_t high = vdupq_n_u8(0x80);
  for (; j+16 <= n; j += 16) {
    const uint8x16_t b = vld1q_u8(reinterpret_cast<const uint8_t*>(begin+j));
    const uint8x16_t eq = vorrq_u8(
        vorrq_u8(vceqq_u8(b, s0), vceqq_u8(b, s1)),
        vorrq_u8(vceqq_u8(b, s2), vcgeq_u8(b, high)));
    // four bits per byte, in text order
    const uint64_t mask = vget_lane_u64(
        vreinterpret_u64_u8(vshrn_n_u16(vreinterpretq_u16_u8(eq), 4)), 0);
    if (mask != 0)
      return j + (__builtin_ctzll(mask) >> 2);
  }
#else
  (void)begin;
  (void)n;
  (void)stop;
#endif
  return j;
}

}  // namespace

class TDFA {
 public:
  // budget: the program's remaining bytes for its automata, charged as
  // states and transitions are built.
  TDFA(Prog* prog, bool endmatch, int ncap, int64_t* budget);
  ~TDFA();

  // 1: matched, 0: no match, -1: past the budget (no result).
  int Search(const StringPiece& text, StringPiece* match, int nmatch);

 private:
  void AddToList(int id0, const int8_t* slots0, std::vector<Thread>* out);
  TState* Intern(const std::vector<Thread>& threads, TEdge* edge);
  uintptr_t Build(TState* s, int cls);
  void AdmitScan(TState* s, uint8_t mode);

  Prog* prog_;
  bool endmatch_;
  int ntrack_;
  int nclass_;
  bool failed_;
  int64_t* budget_;
  std::unordered_map<std::string, TState*> states_;
  std::vector<TEdge*> edges_;
  TState* start_;
  TEdge start_edge_;
  SparseSet seen_;
  std::vector<Thread> stack_;
};

TDFA::TDFA(Prog* prog, bool endmatch, int ncap, int64_t* budget)
    : prog_(prog),
      endmatch_(endmatch),
      ntrack_(ncap > 1 ? 2*(ncap-1) : 0),
      nclass_(prog->bytemap_range()),
      failed_(false),
      budget_(budget),
      start_(NULL),
      seen_(prog->size()) {
  *budget_ -= sizeof(TDFA) + prog->size()*2*sizeof(int);  // seen_
  if (*budget_ < 0) {
    failed_ = true;
    return;
  }
  Thread t;
  t.id = 0;
  for (int k = 0; k < kMaxTrack; k++)
    t.slots[k] = kUnset;
  std::vector<Thread> threads;
  AddToList(prog_->start(), t.slots, &threads);
  if (!failed_)
    start_ = Intern(threads, &start_edge_);
}

TDFA::~TDFA() {
  for (auto& kv : states_)
    DeleteState(kv.second);
  for (TEdge* e : edges_)
    delete e;
}

// NFA::AddToThreadq's walk from id0, without its look at the next byte: a
// ByteRange is a thread whatever it matches (one that cannot match the next
// byte dies on it), and the walk goes on to the rest of its list (whose
// instructions a hint would skip are ByteRanges that cannot match it either).
void TDFA::AddToList(int id0, const int8_t* slots0,
                     std::vector<Thread>* out) {
  Thread f;
  f.id = id0;
  memcpy(f.slots, slots0, sizeof f.slots);
  stack_.clear();
  stack_.push_back(f);
  while (!stack_.empty()) {
    f = stack_.back();
    stack_.pop_back();
    int id = f.id;
  Loop:
    if (id == 0 || seen_.contains(id))
      continue;
    seen_.insert_new(id);
    Prog::Inst* ip = prog_->inst(id);
    switch (ip->opcode()) {
      default:  // not admitted (Prog::EnableTDFA)
        LOG(DFATAL) << "Unexpected opcode in TDFA: " << ip->opcode();
        failed_ = true;
        return;

      case kInstFail:
        break;

      case kInstNop:
        if (!ip->last()) {
          Thread n = f;
          n.id = id+1;
          stack_.push_back(n);
        }
        id = ip->out();
        goto Loop;

      case kInstCapture: {
        if (!ip->last()) {
          Thread n = f;
          n.id = id+1;
          stack_.push_back(n);
        }
        const int j = ip->cap()-2;
        if (0 <= j && j < ntrack_)
          f.slots[j] = kNew;
        id = ip->out();
        goto Loop;
      }

      case kInstByteRange:
      case kInstMatch: {
        Thread t = f;
        t.id = id;
        out->push_back(t);
        if (!ip->last()) {
          id = id+1;
          goto Loop;
        }
        break;
      }
    }
  }
}

// The state for a thread list whose slots are source registers, kNew or
// kUnset; fills edge with the register operations into it.
TState* TDFA::Intern(const std::vector<Thread>& threads, TEdge* edge) {
  size_t n = threads.size();
  int match = -1;
  for (size_t i = 0; i < n; i++) {
    if (prog_->inst(threads[i].id)->opcode() == kInstMatch) {
      match = static_cast<int>(i);
      break;
    }
  }
  // Leftmost-first: a thread after the first Match can only find a worse
  // match.
  if (!endmatch_ && match >= 0)
    n = match+1;

  // Renumber registers in order of first use.
  int8_t map_old[kMaxRegs];   // source register -> new register
  int8_t map_new = kUnset;      // kNew -> new register
  for (int k = 0; k < kMaxRegs; k++)
    map_old[k] = kUnset;
  int nregs = 0;
  std::string key;
  key.reserve(n*(sizeof(int)+ntrack_));
  std::vector<int8_t> slots(n*ntrack_);
  for (size_t i = 0; i < n; i++) {
    key.append(reinterpret_cast<const char*>(&threads[i].id), sizeof(int));
    for (int k = 0; k < ntrack_; k++) {
      const int8_t v = threads[i].slots[k];
      int8_t r = kUnset;
      if ((v == kNew && map_new == kUnset) || (v >= 0 && map_old[v] == kUnset)) {
        if (nregs == kMaxRegs) {
          failed_ = true;
          return NULL;
        }
        edge->src[nregs] = v;
        if (v == kNew)
          map_new = static_cast<int8_t>(nregs);
        else
          map_old[v] = static_cast<int8_t>(nregs);
        nregs++;
      }
      if (v == kNew)
        r = map_new;
      else if (v >= 0)
        r = map_old[v];
      slots[i*ntrack_+k] = r;
    }
    key.append(reinterpret_cast<const char*>(&slots[i*ntrack_]), ntrack_);
  }
  edge->nregs = nregs;

  auto it = states_.find(key);
  if (it != states_.end()) {
    edge->to = it->second;
    return it->second;
  }
  TState* s = NewState(nclass_);
  for (size_t i = 0; i < n; i++)
    s->insts.push_back(threads[i].id);
  s->slots = std::move(slots);
  s->nregs = nregs;
  s->match = match >= 0 && static_cast<size_t>(match) < n ? match : -1;
  s->dead = n == 0;
  // the state, its key (in the map and in insts and slots), next, the map node
  *budget_ -= sizeof(TState) + 2*key.size() + nclass_*sizeof(uintptr_t) + 64;
  if (static_cast<int>(states_.size()) >= kMaxStates || *budget_ < 0)
    failed_ = true;
  states_.emplace(std::move(key), s);
  edge->to = s;
  return s;
}

uintptr_t TDFA::Build(TState* s, int cls) {
  int c = -1;
  const uint8_t* bytemap = prog_->bytemap();
  for (int b = 0; b < 256; b++) {
    if (bytemap[b] == cls) {
      c = b;
      break;
    }
  }
  std::vector<Thread> threads;
  seen_.clear();
  int8_t slots[kMaxTrack];
  for (size_t i = 0; i < s->insts.size(); i++) {
    Prog::Inst* ip = prog_->inst(s->insts[i]);
    if (ip->opcode() != kInstByteRange || !ip->Matches(c))
      continue;
    for (int k = 0; k < ntrack_; k++)
      slots[k] = s->slots[i*ntrack_+k];
    AddToList(ip->out(), slots, &threads);
    if (failed_)
      return 0;
  }
  TEdge* e = new TEdge;
  edges_.push_back(e);
  *budget_ -= sizeof(TEdge) + sizeof(TEdge*);
  TState* to = Intern(threads, e);
  if (to == NULL)
    return 0;
  bool identity = to->nregs == s->nregs;
  for (int k = 0; identity && k < e->nregs; k++)
    identity = e->src[k] == k;
  uintptr_t v = identity ? reinterpret_cast<uintptr_t>(to)
                         : reinterpret_cast<uintptr_t>(e) | 1;
  s->next()[cls] = v;
  if (to == s) {
    // A run of such bytes leaves every register as one step does (kept, or
    // the position after the run's last byte), so it is scanned.
    uint8_t mode = identity ? 1 : 2;
    for (int k = 0; mode == 2 && k < e->nregs; k++)
      if (e->src[k] != k && e->src[k] != kNew)
        mode = 0;
    if (mode == 2) {
      if (s->self_edge == NULL)
        s->self_edge = e;
      else if (memcmp(s->self_edge->src, e->src, e->nregs) != 0)
        mode = 0;
    }
    if (mode != 0)
      for (int b = 0; b < 256; b++)
        if (bytemap[b] == cls)
          s->self[b] = mode;
  }
  return v;
}

// Decides the vector step of s's runs on mode, at the first such run: builds
// every byte class with an ASCII byte, so that self[0..127] is final, and
// takes the step when at most kMaxStops ASCII bytes end the run. A state whose
// builds run past the budget is scanned byte by byte.
void TDFA::AdmitScan(TState* s, uint8_t mode) {
  const uint8_t* bytemap = prog_->bytemap();
  s->scan = kNoScan;
  for (int b = 0; b < 128 && !failed_; b++)
    if (s->next()[bytemap[b]] == 0)
      Build(s, bytemap[b]);
  if (failed_)
    return;
  uint8_t stop[kMaxStops] = {0x80, 0x80, 0x80};
  int nstop = 0;
  for (int b = 0; b < 128; b++) {
    if (s->self[b] == mode)
      continue;
    if (nstop == kMaxStops)
      return;
    stop[nstop++] = static_cast<uint8_t>(b);
  }
  memcpy(s->stop, stop, sizeof stop);
  s->scan = mode;
}

int TDFA::Search(const StringPiece& text, StringPiece* match, int nmatch) {
  if (failed_)
    return -1;
  const char* const begin = text.data();
  const size_t n = text.size();
  const uint8_t* bytemap = prog_->bytemap();
  const bool vector_scan = kVectorScan && prog_->tdfa_vector_scan();
  const char* regs[2][kMaxRegs];
  int cur = 0;
  for (int k = 0; k < start_edge_.nregs; k++)
    regs[cur][k] = begin;
  TState* s = start_;
  bool matched = false;
  const char* mend = NULL;
  const char* mslot[kMaxTrack];
  size_t i = 0;
  for (;;) {
    if (s->match >= 0 && (!endmatch_ || i == n)) {
      matched = true;
      mend = begin+i;
      const int8_t* sl = &s->slots[s->match*ntrack_];
      for (int k = 0; k < ntrack_; k++)
        mslot[k] = sl[k] >= 0 ? regs[cur][sl[k]] : NULL;
    }
    if (i == n || s->dead)
      break;
    int c = static_cast<uint8_t>(begin[i]);
    const uint8_t mode = s->self[c];
    if (mode != 0) {
      // The state steps to itself on these bytes: the registers are what the
      // last step leaves, and a match the state holds is taken again at the
      // end of the run.
      if (vector_scan) {
        if (s->scan == 0)
          AdmitScan(s, mode);
        // the bytes before the first that may end the run continue it
        if (s->scan == mode)
          i = ScanBlocks(begin, i+1, n, s->stop) - 1;
      }
      do {
        i++;
      } while (i < n && s->self[static_cast<uint8_t>(begin[i])] == mode);
      if (mode == 2) {
        const TEdge* e = s->self_edge;
        for (int k = 0; k < e->nregs; k++)
          if (e->src[k] == kNew)
            regs[cur][k] = begin+i;
      }
      continue;
    }
    uintptr_t v = s->next()[bytemap[c]];
    if (v == 0) {
      v = Build(s, bytemap[c]);
      if (failed_)
        return -1;
    }
    i++;
    if (v & 1) {
      const TEdge* e = reinterpret_cast<const TEdge*>(v & ~uintptr_t{1});
      const char* const pos = begin+i;
      for (int k = 0; k < e->nregs; k++)
        regs[1-cur][k] = e->src[k] >= 0 ? regs[cur][e->src[k]] : pos;
      cur = 1-cur;
      s = e->to;
    } else {
      s = reinterpret_cast<TState*>(v);
    }
  }
  if (!matched)
    return 0;
  if (nmatch > 0) {
    match[0] = StringPiece(begin, static_cast<size_t>(mend-begin));
    for (int g = 1; g < nmatch; g++)
      match[g] = StringPiece(mslot[2*g-2],
                             static_cast<size_t>(mslot[2*g-1]-mslot[2*g-2]));
  }
  return 1;
}

// Programs whose instructions are all Fail, Nop, Capture (of a group),
// ByteRange and Match. Call before any search; vector_scan: the scan of a run
// takes 16-byte blocks where the state admits it; thread_private: searches take
// no flag.
void Prog::EnableTDFA(bool vector_scan, bool thread_private) {
  if (reversed_ || tdfa_admitted_)
    return;
  for (int id = 0; id < size_; id++) {
    Inst* ip = inst(id);
    switch (ip->opcode()) {
      case kInstFail:
      case kInstNop:
      case kInstByteRange:
      case kInstMatch:
        break;
      case kInstCapture:
        if (ip->cap() < 2)
          return;
        break;
      default:
        return;
    }
  }
  tdfa_admitted_ = true;
  tdfa_vector_scan_ = vector_scan;
  tdfa_thread_private_ = thread_private;
  tdfa_budget_ = std::min(kMaxBudget, dfa_mem_/4);
  dfa_mem_ -= tdfa_budget_;
}

int Prog::SearchTDFA(const StringPiece& text, const StringPiece& context,
                     MatchKind kind, StringPiece* match, int nmatch) {
  if (!tdfa_admitted_ || kind == kLongestMatch || kind == kManyMatch ||
      nmatch < 0 || nmatch > kMaxTDFACapture)
    return -1;
  // As BitState: an anchored program must be at the ends of the context.
  StringPiece ctx = context.data() == NULL ? text : context;
  if (anchor_start() && ctx.data() != text.data())
    return 0;
  if (anchor_end() && ctx.data()+ctx.size() != text.data()+text.size())
    return 0;
  const bool endmatch = kind == kFullMatch || anchor_end();
  if (tdfa_thread_private_)
    return SearchTDFAThreadPrivate(text, endmatch, match, nmatch);
  // One search at a time; another thread meanwhile runs another engine.
  if (tdfa_busy_.exchange(true, std::memory_order_acquire))
    return -1;
  TDFA*& t = tdfa_[endmatch ? 1 : 0][nmatch];
  if (t == NULL)
    t = new TDFA(this, endmatch, nmatch, &tdfa_budget_);
  const int r = t->Search(text, match, nmatch);
  tdfa_busy_.store(false, std::memory_order_release);
  return r;
}

// The search of a thread-private program: its owner's thread alone searches it,
// so no flag is taken. Out of line, so that path has a symbol of its own.
#if defined(__GNUC__)
__attribute__((noinline))
#endif
int Prog::SearchTDFAThreadPrivate(const StringPiece& text, bool endmatch,
                                  StringPiece* match, int nmatch) {
  TDFA*& t = tdfa_[endmatch ? 1 : 0][nmatch];
  if (t == NULL)
    t = new TDFA(this, endmatch, nmatch, &tdfa_budget_);
  return t->Search(text, match, nmatch);
}

void Prog::DeleteTDFAs() {
  for (auto& row : tdfa_)
    for (TDFA*& t : row) {
      delete t;
      t = NULL;
    }
}

}  // namespace duckdb_re2
