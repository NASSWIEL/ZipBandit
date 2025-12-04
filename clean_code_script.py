import os
import tokenize
import io
import sys

def remove_comments(source):
    io_obj = io.BytesIO(source.encode('utf-8'))
    out = ""
    last_lineno = -1
    last_col = 0
    
    try:
        tokens = tokenize.tokenize(io_obj.readline)
        for tok in tokens:
            token_type = tok.type
            token_string = tok.string
            start_line, start_col = tok.start
            end_line, end_col = tok.end
            
            if start_line > last_lineno:
                last_col = 0
            
            if start_col > last_col:
                out += " " * (start_col - last_col)
            
            if token_type == tokenize.COMMENT:
                pass
            elif token_type == tokenize.ENCODING:
                pass
            else:
                out += token_string
            
            last_col = end_col
            last_lineno = end_line
    except tokenize.TokenError:
        return source
        
    return out

def process_file(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            source = f.read()
        
        new_source = remove_comments(source)
        
        lines = new_source.split('\n')
        cleaned_lines = [line.rstrip() for line in lines]
        
        final_lines = []
        prev_empty = False
        for line in cleaned_lines:
            if not line:
                if not prev_empty:
                    final_lines.append(line)
                    prev_empty = True
            else:
                final_lines.append(line)
                prev_empty = False
                
        final_source = "\n".join(final_lines)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(final_source)
        print(f"Processed {filepath}")
    except Exception as e:
        print(f"Error processing {filepath}: {e}")

if __name__ == "__main__":
    for root, dirs, files in os.walk("."):
        for file in files:
            if file.endswith(".py") and file != "clean_code_script.py":
                path = os.path.join(root, file)
                process_file(path)
