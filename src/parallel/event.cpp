#include "duckdb/parallel/event.hpp"
#include "duckdb/common/assert.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/parallel/task_scheduler.hpp"
#include "duckdb/execution/executor.hpp"
#include "duckdb/parallel/pipeline.hpp"

namespace duckdb {

Event::Event(Executor &executor_p)
    : executor(executor_p), finished_tasks(0), total_tasks(0), finished_dependencies(0), total_dependencies(0),
      finished(false) {
}

void Event::CompleteDependency() {
	idx_t current_finished = ++finished_dependencies;
	D_ASSERT(current_finished <= total_dependencies);
	if (current_finished == total_dependencies) {
		// all dependencies have been completed: schedule the event
		D_ASSERT(total_tasks == 0);
		{
			struct SchedulingGuard {
				bool &scheduling;
				bool saved;
				~SchedulingGuard() {
					scheduling = saved;
				}
			} guard {Pipeline::SchedulingCompletedDependency(), Pipeline::SchedulingCompletedDependency()};
			guard.scheduling = true;
			Schedule();
		}
		if (total_tasks == 0) {
			Finish();
		}
	}
}

void Event::Finish() {
	D_ASSERT(!finished);
	FinishEvent();
	finished = true;
	// finished processing the pipeline, now we can schedule pipelines that depend on this pipeline
	for (auto &parent_entry : parents) {
		auto parent = parent_entry.lock();
		if (!parent) { // LCOV_EXCL_START
			continue;
		} // LCOV_EXCL_STOP
		// mark a dependency as completed for each of the parents
		parent->CompleteDependency();
	}
	FinalizeFinish();
}

void Event::AddDependency(Event &event) {
	total_dependencies++;
	event.parents.push_back(weak_ptr<Event>(shared_from_this()));
#ifdef DEBUG
	event.parents_raw.push_back(*this);
#endif
}

const vector<reference<Event>> &Event::GetParentsVerification() const {
	D_ASSERT(parents.size() == parents_raw.size());
	return parents_raw;
}

void Event::FinishTask() {
	D_ASSERT(finished_tasks.load() < total_tasks.load());
	idx_t current_tasks = total_tasks;
	idx_t current_finished = ++finished_tasks;
	D_ASSERT(current_finished <= current_tasks);
	if (current_finished == current_tasks) {
		Finish();
	}
}

ClientContext &Event::GetClientContext() {
	return executor.context;
}

void Event::InsertEvent(shared_ptr<Event> replacement_event) {
	replacement_event->parents = std::move(parents);
#ifdef DEBUG
	replacement_event->parents_raw = std::move(parents_raw);
#endif
	replacement_event->AddDependency(*this);
	executor.AddEvent(std::move(replacement_event));
}

void Event::SetTasks(vector<shared_ptr<Task>> tasks) {
	auto &ts = TaskScheduler::GetScheduler(executor.context);
	D_ASSERT(total_tasks == 0);
	D_ASSERT(!tasks.empty());
	this->total_tasks = tasks.size();
	auto &inline_requested = Pipeline::InlineTaskRequested();
	if (inline_requested && tasks.size() == 1) {
		// run the one task of a sequential pipeline here for a partial step instead of queueing it and
		// waking a worker; what it does not finish takes the usual paths (the queue, or the executor's reschedule set)
		inline_requested = false;
		auto task = std::move(tasks[0]);
		auto result = task->Execute(TaskExecutionMode::PROCESS_PARTIAL);
		switch (result) {
		case TaskExecutionResult::TASK_FINISHED:
		case TaskExecutionResult::TASK_ERROR: // the executor holds the error, as for a queued task
			return;
		case TaskExecutionResult::TASK_NOT_FINISHED:
			ts.ScheduleTask(executor.GetToken(), std::move(task));
			return;
		case TaskExecutionResult::TASK_BLOCKED:
			task->Deschedule();
			return;
		}
		return;
	}
	ts.ScheduleTasks(executor.GetToken(), tasks);
}

} // namespace duckdb
