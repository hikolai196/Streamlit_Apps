import pandas as pd
import matplotlib.pyplot as plt


def execute_code(code, df):
    local_vars = {
        "df": df,
        "pd": pd,
        "plt": plt,
    }

    try:
        exec(code, {"__builtins__": {}}, local_vars)
        return local_vars.get("result", None), None
    except Exception as e:
        return None, str(e)
