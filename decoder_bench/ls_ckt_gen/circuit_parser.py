import numpy as np
from pyparsing import Word, alphas, nums, delimitedList, quotedString

# Parse a circuit file/string that contains information on logical qubits and 
# logical operations. 

# OpenQASM is the chosen framework to parse. This is a barebones parser, more
# complicated parsers and compilers can be built on top of this for the simulator. 

# TODO: parse the circuit and identify dependencies and annotate them so that
# parallel gates can be identified.

class circuit_parser():
    def __init__(self) -> None:
        identifier = Word(alphas, alphas + nums + "_").setName('identifier')
        integer = Word(nums)
        string = quotedString
        argument2 = delimitedList(identifier + "[" + integer + "]")
        argument1 = (identifier + "[" + integer + "]")
        measure_all = identifier + "->" + identifier
        measure = argument1 + "->" + argument1
        line = identifier + (identifier ^ argument1 ^ argument2 ^ integer ^ string ^ measure ^ measure_all)
        self.program = delimitedList(line, delim=';')
        self.cmds = []
        return

    def parse(self, ckt:str) -> dict:
        ckt = ckt.replace(";", ";\n") # add newlines after every semicolon
        for l in ckt.split('\n'):
            if l.strip():
                self.cmds.append(self.program.parse_string(l))
        # Find number of logical qubits specified
        ids = [cmd[0] for cmd in self.cmds]
        idx = ids.index('qreg')
        cmd = self.cmds[idx]
        # qreg: ['qreg', 'qreg_name', '[', 'num_qubits', ']']
        num_qubits = int(cmd[-2])

        directives_grammar = ['include', 'OPENQASM', 'qreg', 'creg', 'barrier', 'measure']
        gates_grammar = ['H', 'CX', 'X', 'I']
        grammar = directives_grammar + gates_grammar
        cmds = {i:[] for i in range(num_qubits)}
        # Iterate through cmds
        for cmd in self.cmds:
            if cmd[0] not in grammar:
                raise ValueError('%s is not supported'%(cmd[0]))
            if cmd[0] in gates_grammar:
                if cmd[0] == 'CX':
                    cmd[3] = int(cmd[3])
                    cmd[7] = int(cmd[7])
                    entry = tuple((cmd[0], cmd[3], cmd[7]))
                    temp = list(cmds.keys())
                    temp.remove(cmd[3])
                    temp.remove(cmd[7])
                    cmds[cmd[3]].append(entry)
                    cmds[cmd[7]].append(entry)
                    # Append None for every other qubit
                    for q in temp:
                        cmds[q].append(None)
                else:
                    cmd[3] = int(cmd[3])
                    entry = tuple((cmd[0], cmd[3]))
                    cmds[cmd[3]].append(entry)
                    temp = list(cmds.keys())
                    temp.remove(cmd[3])
                    for q in temp:
                        cmds[q].append(None)
                pass
            pass
        return cmds
    
    def from_file(self, file:str) -> dict:
        with open(file, 'r') as f:
            ckt = f.read()
        cmds = self.parse(ckt)
        return cmds
    
    def from_string(self, ckt:str) -> dict:
        assert(type(ckt) == str)
        # Circuit specified as a string
        cmds = self.parse(ckt)
        return cmds

    pass # circuit_parser()

if __name__ == '__main__':
    p = circuit_parser().from_string(ckt='''
        OPENQASM 2.0;
        include "qelibc";
        qreg qubits[10];
        CX q[0], q[1];
        barrier q[0], q[1];
        H q[0];
        measure q[0] -> c[0];
    ''')
    print(p)