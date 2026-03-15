import sys
sys.path.append(r'e:\Coding\TRM Material Science\matbench_phonons')
import importlib.util
spec = importlib.util.spec_from_file_location("main", r"e:\Coding\TRM Material Science\matbench_phonons\phonons_v6 new.py")
foo = importlib.util.module_from_spec(spec)
sys.modules["main"] = foo
spec.loader.exec_module(foo)

print('D=64: ', foo.PhononV6(361, 15, d=64).count_parameters())
print('D=68: ', foo.PhononV6(361, 15, d=68).count_parameters())
print('D=72: ', foo.PhononV6(361, 15, d=72).count_parameters())
print('D=80: ', foo.PhononV6(361, 15, d=80).count_parameters())
