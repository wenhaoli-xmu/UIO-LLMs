def get_modifier(method: str, model_type):
    if method == 'ccf':
        if model_type == 'llama':
            from src.modifiers.modify_llama_ccf import CCF, Teacher

            return Teacher, CCF

