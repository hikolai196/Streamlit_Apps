from eda_agent.execution.executor import execute_code
from eda_agent.llm.client import generate_code


def run_agent(user_input, df, history):

    # 1️⃣ 產生 code
    code = generate_code(user_input, history)

    # 2️⃣ 執行
    result, error = execute_code(code, df)

    # 3️⃣ retry（簡單版）
    if error:
        fix_prompt = f"""
剛剛這段 code 出錯：

{code}

錯誤：
{error}

請修正，重新輸出完整 code。
"""

        code = generate_code(fix_prompt, history)
        result, error = execute_code(code, df)

    return code, result, error
