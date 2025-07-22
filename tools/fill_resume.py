import subprocess
import os
from jinja2 import Environment, FileSystemLoader
import shutil 
import glob
import re


def escape_latex_special_chars(text):
    """
    Escapes special LaTeX characters in a given string.
    """
    # Order matters here. Backslash must be first.
    replacements = {
        '\\': r'\textbackslash{}',
        '&': r'\&',
        '%': r'\%',
        '$': r'\$',
        '#': r'\#',
        '_': r'\_',
        '{': r'\{',
        '}': r'\}',
        '~': r'\textasciitilde{}',
        '^': r'\textasciicircum{}',
    }
    # Create a regex pattern to find all special characters
    # This is more efficient than calling replace() multiple times
    regex = re.compile('([&%$#_{}])'.replace('{}', '\\' + '\\'.join(replacements.keys())))
    
    # Use a lambda function for the replacement
    return regex.sub(lambda mo: replacements[mo.string[mo.start():mo.end()]], text)


def clean_output_folder():
    for ext in (".aux", ".log", ".pdf", ".toc",".synctex.gz"):
        for f in glob.glob(os.path.join("output", f"resume{ext}")):
            os.remove(f)

def render_tex(data, template_dir="templates", template_name="resume_template.tex"):
    env = Environment(
        loader=FileSystemLoader(template_dir),
        block_start_string='{%',
        block_end_string='%}',
        variable_start_string='{{',
        variable_end_string='}}',
        comment_start_string='##',
        comment_end_string='##',
        autoescape=False,
    )

    env.filters['escape'] = escape_latex_special_chars
    template = env.get_template(template_name)
    rendered = template.render(**data)
    out_tex = os.path.join("output", "resume.tex")
    os.makedirs("output", exist_ok=True)
    with open(out_tex, "w", encoding='utf-8') as f:
        f.write(rendered)
    return out_tex

def compile_tex(tex_path):
    clean_output_folder()
    cwd, fname = os.path.split(tex_path)
   # copy_class_file()
    process = subprocess.run(["xelatex", "-interaction=nonstopmode", fname], cwd=cwd, check=True, text=True, capture_output=True)
    if process.returncode != 0:
        print("LaTeX compilation failed!")
        print(process.stdout)
        print(process.stderr)
        return None

    os.path.join(cwd, fname.replace(".tex", ".pdf"))
    return fname.replace(".tex", ".pdf")

