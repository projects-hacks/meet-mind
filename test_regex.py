import re
import json

text = '{"action": "provide_insight", "params": {"insight": "Consider using a hybrid approach: Use a SQL database for structured data"'

name = "provide_insight"

salvaged = re.sub(r'(?i)"?action"?\s*:\s*"?' + name + r'"?,?', '', text)
salvaged = re.sub(r'(?i)"?(params|insight|suggestion|arguments|reasoning|task)"?\s*:\s*', '', salvaged)
salvaged = re.sub(r'[\{\}\[\]]', '', salvaged)  # Strip all curlies/brackets
salvaged = salvaged.replace('""', '"').strip(' ;,\n')

if salvaged.startswith('"') and salvaged.endswith('"'):
    salvaged = salvaged[1:-1]
if salvaged.startswith("'") and salvaged.endswith("'"):
    salvaged = salvaged[1:-1]
salvaged = salvaged.strip()

print(salvaged)
