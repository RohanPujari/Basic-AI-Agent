# Agentic AI Playground — Project Notes

## Goal and learning preference

Finish a small, interview-explainable Weather + Calculator Agentic AI Playground by Sunday, October 25, 2026. The learner already knows the concepts at a theory level and learns fastest by seeing diagrams, implementing each concept, and running small examples. Teach one concept at a time; explain what changed and why before moving on.

After this project, start a separate practical n8n learning project the following week. Include a beginner-friendly introduction to MLOps and deployment in the Weather + Calculator project.

## Existing project and resume point

- Keep the original `app.py`, `tools.py`, and LangGraph learning examples intact.
- Use `agentic_playground.py` as the incremental learning implementation.
- Step 1: Tool calling — present as `run_tool_calling()` (one tool round).
- Step 2: ReAct — present as `run_react()` (bounded repeated tool decisions and visible action/observation trace).
- Step 3: Explicit per-request state — present as one plain Python dictionary holding the request, message history, observations, step count, and final answer.
- Step 4: Planning — present as `create_plan()`; the plan is saved in state and added to the conversation before tool use.
- Step 5: Routing — `route_request()` selects weather, calculator, both, or general; `tools_for_route()` exposes only the relevant tools.
- Step 6: Reflection — `reflect_on_answer()` checks a draft against the request and tool results before returning it.
- Step 7: Multi-agent + supervisor — `run_weather_agent()` and `run_calculator_agent()` are focused specialists; `run_supervisor()` chooses one or coordinates both in order.
- Step 8: Human-in-the-loop — `ask_human_to_approve()` pauses before accepting the final answer.
- Step 9: Evaluation — `evaluate_agent.py` checks routing, calculator output, and weather-to-calculator delegation with offline examples.
- Step 10: Security — enforce the route's tool allowlist at execution time and validate input sizes and types.
- Step 11: Lightweight production structure and MLOps — direct dependencies are pinned, setup is documented, and GitHub Actions runs the offline evaluations on changes.
- Next: Step 12, deployment.

## Code teaching style

Keep the main agent flow in one beginner-readable function. Use short comments, plain data structures, and make the message roles (`system`, `user`, `assistant`, `tool`) visible. Explain each section in simple language before adding another concept. The restricted calculator parser is a separate safety detail; do not let it obscure the tool-calling/ReAct/state lesson.
- Weather currently returns clearly labeled sample data, not live forecasts.
- The calculator uses a restricted arithmetic parser rather than `eval`.
- `requirements.txt` contains the project dependencies; setup and deployment notes still need to be added.

## Sunday build sequence

1. State: explicit request, messages, observations, and completed step count.
2. Planning: turn a multi-part request into a short list of tasks.
3. Routing: direct each task to weather or calculator capability.
4. Reflection: check tool observations/results and recover from an error once, within limits.
5. Multi-agent + supervisor: weather and calculator specialists coordinated by one small supervisor; avoid unnecessary framework complexity.
6. Human-in-the-loop: pause for approval before continuing a clearly marked approval-gated action.
7. Evaluation: a small offline example set checks tool choice, result correctness, failures, and step limits.
8. Security: allowlisted tools, validated inputs, restricted calculator, secrets in environment, and bounded iterations.
9. Lightweight production + MLOps: dependencies and setup instructions, structured logs, reproducible evaluation, a small CI check, and a simple deployable interface.
10. Deployment: run locally first, then deploy the minimal app using a beginner-friendly host and document required secrets.

## Teaching loop

For each step: show a small diagram and concrete request, implement only that step, run a focused check, explain the code path, and keep a short note of what the learner can explain in an interview. Sunday is the only free day this week (Saturday is work); prioritize a working, understandable MVP over advanced infrastructure.

## LinkedIn post idea to revisit

Explain an agent's short-term message history as a working notepad: it starts with the system instructions and user request, records the assistant's tool call, records the tool's result, and sends the updated history back to the model so it can decide what to do next or answer. Emphasize that this is per-run working state, not permanent memory. Create a simple visual and one LinkedIn post after the Weather + Calculator project is complete.
