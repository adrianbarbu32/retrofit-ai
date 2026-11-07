# Demo video script (target 3:30, narrated screen recording)

**Setup before recording:** `streamlit run app/console.py`, browser full screen, zoom 90 %. Close other tabs. Pause
between scenes and cut later. Speak slowly; numbers come from `results/metrics_table.md`.

| Time | Screen | Narration (suggested) |
|---|---|---|
| 0:00-0:20 | Title slide / console header | "Ships and factories run pumps that are 10-25 years old. Operators find out about wear when a fixed alarm limit is crossed, often too late to plan. Replacing a working pump or its controls to add AI is expensive. RETROFIT AI is a listen-only intelligence layer for the machines that are already there." |
| 0:20-0:45 | Architecture slide | "For every minute, our healthy-pump model computes what this exact pump, from its datasheet, would read if it were healthy, at today's duty, inlet temperature and run state. The difference is the evidence. Thirteen fault models, trained on 500 simulated pump-weeks we generated with the organizers' simulator, name the cause. A cautious policy raises one alert per problem." |
| 0:45-1:30 | Console - PUMP-08 (validation, wireless retrofit). Drag the replay clock from 0 h to the first alert | "This seawater pump got a cheap wireless vibration sensor and a suction transmitter. Replaying its week: here the suction pressure starts falling below the healthy model, and discharge pressure and motor power fall with it: less water is getting through. The assistant says *suction strainer clogging*, with the evidence in kPa and percent, a confidence, and a recommendation for a human: clean the strainer at the next opportunity. That is many hours before the cooling-flow limit." |
| 1:30-1:55 | What-if tab | "How bad is it? Fitting the physics to the readings estimates the clog level. The sandbox, simulation only, shows that cleaning restores the flow and the NPSH margin. Nothing is sent to the pump; it is a decision aid." |
| 1:55-2:15 | PUMP-09, sensor alert card (stuck / drift) | "Not every alarm is the machine. Here the bearing temperature reading froze while load changed. The physically coupled signals still agree with the healthy model, so the assistant says *instrument fault: check the transmitter, no machine intervention*. This avoids a needless repair." |
| 2:15-2:30 | Acknowledge / Request work order, fleet tab log | "Operators acknowledge, reject a false alarm, or turn an alert into a work-order request for the planner. Every decision is logged for the shadow-mode review." |
| 2:30-3:00 | Results slide | "Measured with the official scorer: on the validation pumps we score 98 out of 100, against 78 for the best baseline. Every fault is named with the right cause, against 44 percent for the baseline, with zero false alerts, and on median 16 hours before the conventional limit is reached. On 75 unseen simulated pumps we score 89 against 56. And on identical pumps with different instrumentation, adding one suction transmitter and one wireless vibration point raises the share of faults caught with the right cause from 56 to 80 percent." |
| 3:00-3:20 | Edge / integration slide | "It runs on a EUR 500 edge box, offline, reading OPC UA or Modbus. There is no write path. The same three layers transfer to fans and compressors: swap the physics model, keep the rest." |
| 3:20-3:30 | Limits slide | "Limits: synthetic data, simulator-to-simulator results, advisory only, no certification. The next step is a shadow-mode pilot on a real pump." |

**Fallback if the live console is slow:** use the screenshots in `docs/figures/`.
