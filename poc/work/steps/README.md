# Step-Based RCS Automation

Each file in this folder is executable by itself, so you can stop at any point and test only one step.

## Standalone steps

- Login:
  - `python poc/work/steps/login.py`
- Switch tab:
  - `python poc/work/steps/switch_tab.py`
- List tools:
  - `python poc/work/steps/list_tools.py`
- Select tool:
  - `python poc/work/steps/select_tool.py`

## Optional orchestrator

- Run the sequence defined in constants:
  - `python poc/work/steps/run_steps.py`

## Notes

- Step scripts are thin wrappers around existing `poc/work/*.py` scripts, so current behavior is preserved.
- No CLI options are required for step scripts.
- Change behavior by editing constants at the top of each script:
  - `RUN_LOGIN_FIRST`
  - `SWITCH_TO_LIST_FIRST`
  - `TARGET_TAB`
  - `TOOL_NAME`
  - `DOUBLE_CLICK`
