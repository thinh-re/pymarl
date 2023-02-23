import json
import yaml

import dataclass_wizard.wizard_cli as cli

with open('./src/config/combined_env.yaml', 'r') as f:
    data = yaml.safe_load(f)

print(cli.PyCodeGenerator(file_contents=json.dumps(data), experimental=True).py_code)