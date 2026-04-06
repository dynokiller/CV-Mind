import re
import sys

def process_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        code = f.read()

    # Regex to match: any_collection.find_one(...)
    # We use a simple regex that finds the '(' and we manually balance it to find the end ')'
    def replace_call(match, func_name):
        prefix = match.group(1) # collection name
        start = match.end()
        # count parens to find the closing one
        count = 1
        end = start
        while count > 0 and end < len(code):
            if code[end] == '(': count += 1
            elif code[end] == ')': count -= 1
            end += 1
        
        args = code[start:end-1] # content inside parens
        return f"{func_name}({prefix}, {args})"

    # Iterative replacement for find_one
    while True:
        match = re.search(r'([a-zA-Z0-9_]+_collection)\.find_one\(', code)
        if not match: break
        replaced = replace_call(match, "safe_find_one")
        code = code[:match.start()] + replaced + code[match.start() + len(match.group(0)) + len(replaced) - len("safe_find_one(" + match.group(1) + ", "):]
        
    """
    Wait, the above while True loop replacement logic is flawed.
    Let's use a simpler approach.
    """
    
    parts = code.split('.find_one(')
    out_code = parts[0]
    for part in parts[1:]:
        # find the collection name backwards from out_code
        # it will be something like ` user_collection` or `existing = stats_collection`
        import string
        idx = len(out_code) - 1
        while idx >= 0 and (out_code[idx] in string.ascii_letters + string.digits + '_'):
            idx -= 1
        coll_name = out_code[idx+1:]
        out_code = out_code[:idx+1] + "safe_find_one(" + coll_name + ", " + part
    code = out_code
    
    parts = code.split('.insert_one(')
    out_code = parts[0]
    for part in parts[1:]:
        import string
        idx = len(out_code) - 1
        while idx >= 0 and (out_code[idx] in string.ascii_letters + string.digits + '_'):
            idx -= 1
        coll_name = out_code[idx+1:]
        out_code = out_code[:idx+1] + "safe_insert(" + coll_name + ", " + part
    code = out_code

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(code)

if __name__ == "__main__":
    process_file("app.py")
