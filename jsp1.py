# https://chat.openai.com/c/36bbe9e7-7226-4826-8cd9-01d2ec039e72 - 2/3/2024

import numpy as np

class JobShopScheduling:
    def __init__(self, machines, jobs):
        self.machines = machines
        self.jobs = jobs
        self.num_jobs = len(jobs)
        self.num_machines = machines
        self.best_schedule = None
        self.best_makespan = float('inf')

    def is_feasible(self, schedule, job, machine, current_time):
        for i in range(machine):
            if schedule[job, i] + self.jobs[job][i] > current_time:
                return False
        for j in range(job):
            if schedule[j, machine] + self.jobs[j][machine] > current_time:
                return False
        return True

    def calculate_makespan(self, schedule):
        return np.max(np.sum(schedule, axis=0))

    def branch_and_bound(self, schedule, job, current_time):
        if job == self.num_jobs:
            makespan = self.calculate_makespan(schedule)
            if makespan < self.best_makespan:
                self.best_makespan = makespan
                self.best_schedule = schedule.copy()
            return

        for machine in range(self.num_machines):
            if self.is_feasible(schedule, job, machine, current_time):
                schedule[job, machine] = current_time
                next_time = max(current_time, schedule[job, machine]) + self.jobs[job][machine]
                self.branch_and_bound(schedule, job + 1, next_time)
                schedule[job, machine] = 0

    def solve(self):
        initial_schedule = np.zeros((self.num_jobs, self.num_machines), dtype=int)
        self.branch_and_bound(initial_schedule, 0, 0)
        return self.best_schedule, self.best_makespan


# Example usage:
machines = 3
jobs = np.array([[2, 1, 4], [3, 2, 1], [1, 4, 3]])

scheduler = JobShopScheduling(machines, jobs)
best_schedule, makespan = scheduler.solve()

print("Best Schedule:")
print(best_schedule)
print("Makespan:", makespan)
