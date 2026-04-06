import re
import sys
import string

def process_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        code = f.read()

    # Find the position of 'is_logged_in()' to avoid replacing the safe helpers block
    idx_start = code.find('def is_logged_in()')
    if idx_start == -1: return
    
    header = code[:idx_start]
    body = code[idx_start:]

    def replace_method(text, method_name, wrapper_name):
        parts = text.split('.' + method_name + '(')
        out_code = parts[0]
        for part in parts[1:]:
            idx = len(out_code) - 1
            while idx >= 0 and (out_code[idx] in string.ascii_letters + string.digits + '_'):
                idx -= 1
            coll_name = out_code[idx+1:]
            
            # special case: if coll_name is empty or not a collection, skip?
            if "_collection" not in coll_name:
                out_code = out_code + '.' + method_name + '(' + part
            else:
                out_code = out_code[:idx+1] + wrapper_name + "(" + coll_name + ", " + part
        return out_code

    body = replace_method(body, 'update_one', 'safe_update_one')
    body = replace_method(body, 'delete_many', 'safe_delete_many')
    body = replace_method(body, 'find', 'safe_find')
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(header + body)

if __name__ == "__main__":
    process_file("app.py")
