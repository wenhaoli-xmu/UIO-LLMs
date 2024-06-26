def get_modifier(method: str, model_type):
    if method == 'uiollms':
        if model_type == 'llama':
            from src.modifiers.modify_llama_uiollms import UIOLLMs, Teacher
            return Teacher, UIOLLMs