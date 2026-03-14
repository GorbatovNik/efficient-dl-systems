import json
import time
import torch
import os
from collections import defaultdict


class Profile:
    def __init__(self, model, name="model", schedule=None):
        self.name_map = self._build_name_map(model, name)
        self.events = []
        self.model = model
        self.hooks = []

        if schedule is None:
            self.schedule = None
            self.active = True
        else:
            self.schedule = schedule
            self.active = False

        self.step_num = 0
        self._current_phase = None
        self._phase_steps = []
        self._setup_schedule()

        self._forward_start_events = {}
        self._forward_end_events = {}
        self._backward_start_events = {}
        self._backward_end_events = {}

        self._start_time = None

        self._use_cuda = torch.cuda.is_available()

    def _setup_schedule(self):
        if self.schedule is None:
            self.active = True
            return

        self._phase_steps = []
        cumulative = 0
        for phase, count in self.schedule:
            for i in range(count):
                self._phase_steps.append(phase)
            cumulative += count
        self._update_phase()

    def _update_phase(self):
        if self.schedule is None:
            self.active = True
            return

        if self.step_num < len(self._phase_steps):
            self._current_phase = self._phase_steps[self.step_num]
            self.active = (self._current_phase == "active")
        else:
            self.active = False

    def _build_name_map(self, model, name="model"):
        name_map = {}
        for full_name, module in model.named_modules():
            if full_name == "":
                full_name = name

            if self._is_leaf(module):
                name_map[module] = module.__class__.__name__
            else:
                name_map[module] = f"{full_name}: {module.__class__.__name__}"

        return name_map

    def _is_leaf(self, module):
        return len(list(module.children())) == 0

    def _get_time_us(self):
        if self._use_cuda:
            torch.cuda.synchronize()
        return time.perf_counter() * 1e6

    def _forward_pre_hook(self, module, inputs):
        if not self.active:
            return
        if self._use_cuda:
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            self._forward_start_events[module] = event
        else:
            self._forward_start_events[module] = time.perf_counter()

    def _forward_post_hook(self, module, inputs, outputs):
        if not self.active:
            return
        if module not in self._forward_start_events:
            return
        if self._use_cuda:
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            self._forward_end_events[module] = event
        else:
            end_time = time.perf_counter()
            start_time = self._forward_start_events.pop(module)
            dur_us = (end_time - start_time) * 1e6
            self.events.append({
                "name": self.name_map.get(module, str(module)),
                "ph": "X",
                "ts": (start_time - self._start_time) * 1e6,
                "dur": dur_us,
                "cat": "forward",
                "pid": 0,
                "tid": 0,
            })

    def _backward_pre_hook(self, module, grad_output):
        if not self.active:
            return
        if self._use_cuda:
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            self._backward_start_events[module] = event
        else:
            self._backward_start_events[module] = time.perf_counter()

    def _backward_post_hook(self, module, grad_input, grad_output):
        if not self.active:
            return
        if module not in self._backward_start_events:
            return
        if self._use_cuda:
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            self._backward_end_events[module] = event
        else:
            end_time = time.perf_counter()
            start_time = self._backward_start_events.pop(module)
            dur_us = (end_time - start_time) * 1e6
            self.events.append({
                "name": self.name_map.get(module, str(module)),
                "ph": "X",
                "ts": (start_time - self._start_time) * 1e6,
                "dur": dur_us,
                "cat": "backward",
                "pid": 0,
                "tid": 1,
            })

    def _flush_cuda_events(self):
        if not self._use_cuda:
            return

        torch.cuda.synchronize()

        for module in list(self._forward_start_events.keys()):
            if module in self._forward_end_events:
                start_event = self._forward_start_events.pop(module)
                end_event = self._forward_end_events.pop(module)
                dur_ms = start_event.elapsed_time(end_event)
                dur_us = dur_ms * 1000
                self.events.append({
                    "name": self.name_map.get(module, str(module)),
                    "ph": "X",
                    "ts": 0,
                    "dur": dur_us,
                    "cat": "forward",
                    "pid": 0,
                    "tid": 0,
                })

        for module in list(self._backward_start_events.keys()):
            if module in self._backward_end_events:
                start_event = self._backward_start_events.pop(module)
                end_event = self._backward_end_events.pop(module)
                dur_ms = start_event.elapsed_time(end_event)
                dur_us = dur_ms * 1000
                self.events.append({
                    "name": self.name_map.get(module, str(module)),
                    "ph": "X",
                    "ts": 0,
                    "dur": dur_us,
                    "cat": "backward",
                    "pid": 0,
                    "tid": 1,
                })

    def __enter__(self):
        self._start_time = time.perf_counter()
        self.events = []

        for module in self.model.modules():
            handle = module.register_forward_pre_hook(self._forward_pre_hook)
            self.hooks.append(handle)
            handle = module.register_forward_hook(self._forward_post_hook)
            self.hooks.append(handle)
            handle = module.register_full_backward_pre_hook(self._backward_pre_hook)
            self.hooks.append(handle)
            handle = module.register_full_backward_hook(self._backward_post_hook)
            self.hooks.append(handle)

        return self

    def __exit__(self, type, value, traceback):
        self._flush_cuda_events()
        for handle in self.hooks:
            handle.remove()
        self.hooks = []

    def step(self):
        if self._use_cuda:
            self._flush_cuda_events()

        self.step_num += 1
        self._update_phase()

    def summary(self):
        print("Summary:")
        stats = defaultdict(lambda: {"count": 0, "total_us": 0.0})
        for event in self.events:
            key = (event["name"], event["cat"])
            stats[key]["count"] += 1
            stats[key]["total_us"] += event["dur"]

        for (name, cat), s in sorted(stats.items(), key=lambda x: -x[1]["total_us"]):
            avg_us = s["total_us"] / s["count"] if s["count"] > 0 else 0
            print(f"  [{cat}] {name}: count={s['count']}, "
                  f"total={s['total_us']:.1f}us, avg={avg_us:.1f}us")

    def to_perfetto(self, path="trace.json"):
        ts_cursor = 0.0
        trace_events = []
        for event in self.events:
            ev = dict(event)
            if ev["ts"] == 0:
                ev["ts"] = ts_cursor
                ts_cursor += ev["dur"]
            trace_events.append(ev)

        trace_events.insert(0, {
            "name": "process_name",
            "ph": "M",
            "pid": 0,
            "tid": 0,
            "args": {"name": "Model"},
        })
        trace_events.insert(1, {
            "name": "thread_name",
            "ph": "M",
            "pid": 0,
            "tid": 0,
            "args": {"name": "Forward"},
        })
        trace_events.insert(2, {
            "name": "thread_name",
            "ph": "M",
            "pid": 0,
            "tid": 1,
            "args": {"name": "Backward"},
        })

        with open(path, "w") as f:
            json.dump({"traceEvents": trace_events}, f, indent=2)

        print(f"Trace saved to {path}")
