import numpy as np
import pandas as pd
import os.path


class ImpedanceHero():
    def __init__(self, itype="fri") -> None:
        self.itype = itype

    def show_status(self):
        pass

    def set_type(self, itype):
        self.itype = itype

    def loadfile(self, fname, ftype="JiangLabVIEW"):

        if ftype == "JiangLabVIEW":  # Jiang LabVIEW format
            self.data = self.load_jiang(fname)
        elif ftype == "fma":  # freq, magnitude, angle
            self.data = self.load_fma(fname)
        elif ftype == "fri":  # freq, real, imaginary
            self.data = self.load_fri(fname)

        if self.itype == "fma":
            self.fri2fma()

    def writefile(self, dest, ftype="csv"):
        if ftype == "csv":
            pd.DataFrame(self.data).to_csv(dest, index=None, header=None)

    def load_jiang(self, fname):
        imp_data = pd.read_excel(
            fname, index_col=None, engine="openpyxl").to_numpy()

        imp_data = np.reshape(imp_data, [imp_data.shape[0], len(self.frequencies), 5])[
            :, :, 2:4]
        imp_data[imp_data == 0] = np.nan
        imp_data[imp_data > 1e30] = np.nan

        processed_data = self.process_jiang(imp_data, self.frequencies)

        return processed_data

    def load_fma(self, df):
        pass

    def load_fri(self, df):
        pass

    def process_jiang(self, raw_data, frequencies):     # generated by chatGPT
        num_frequencies = raw_data.shape[1]
        num_measurements = raw_data.shape[0]
        frequencies = np.repeat(frequencies, num_measurements)
        processed_data = np.zeros((num_frequencies * num_measurements, 3))
        processed_data[:, 0] = frequencies
        processed_data[:, 1:] = raw_data.reshape(-1, 2)
        return processed_data

    def loadfreq_jiang(self, fname):
        with open(fname) as f:
            for line in f:
                if line.split(': ')[0] == 'Signal Frequency (Hz)':
                    flist = line.split(': ')[1].split('.000000')
                    flist.pop()
                    flist = [float(i) for i in flist]
                    # print(flist)
                    self.frequencies = flist
                    return flist

    def fri2fma(self):
        '''converts self.data from fri to fma'''
        imp_cplx = self.data[:, 1]+1j*self.data[:, 2]
        imp_mag = np.abs(imp_cplx)
        imp_ang = np.angle(imp_cplx, deg=True)
        self.data[:, 1] = imp_mag
        self.data[:, 2] = imp_ang

    def build_model(self, circuit, name):
        def parse_circuit(circuit, count=-1):
            if type(circuit) == list:
                impedance = "( "
                for subcircuit in circuit:
                    new_impedance, count = parse_circuit(subcircuit, count)
                    impedance += new_impedance
                    impedance += " + "
                impedance = impedance[:-2]
                impedance += ")"
                return impedance, count

            elif type(circuit) == tuple:
                impedance = "1 / ( "
                for subcircuit in circuit:
                    impedance += "1 / ( "
                    new_impedance, count = parse_circuit(subcircuit, count)
                    impedance += new_impedance
                    impedance += " ) + "
                impedance = impedance[:-2]
                impedance += ")"
                return impedance, count

            elif circuit == "C":
                count += 1
                return f"1 / (2j * np.pi * f * params[{count}])", count

            elif circuit == "R":
                count += 1
                return f"params[{count}]", count

            elif circuit == "L":
                count += 1
                return f"2j * np.pi * f * params[{count}]", count

            elif circuit == "E":
                count += 2
                return f"1 / ( params[{count-1}] * ((2j * np.pi * f) ** params[{count}]))", count

            elif type(circuit) == str:
                raise ValueError(f"Unsupported element: "+circuit)
            else:
                raise TypeError(f"Unsupported impedance type: {type(circuit)}")

        if not os.path.isfile("eq_models.py"):
            with open("eq_models.py", "w") as file:
                file.write("import numpy as np\n\n")
                file.close()

        with open("eq_models.py", "r") as file:
            if name in self.list_models():
                raise NameError("Model name already exists!")
            print("Model name is available. Proceed to create new model")
            file.close()

        with open("eq_models.py", "a") as file:
            file.write("\n")
            file.write("def "+name+"(f, params):\n")
            file.write("    return " + parse_circuit(circuit)[0])
            file.close()

    def list_models(self):
        mlist=[]
        with open("eq_models.py", "r") as file:
            for line in file.readlines():
                if line.startswith("def "):
                    mlist.append(line.split(" ")[1].split("(")[0])
            file.close()
        return mlist

        # elif circuit.key()=="C":
        #     return 1/(2j*np.pi*freq*circuit.value())
        # elif circuit.key()=="R":
        #     return circuit.value()
        # elif circuit.key()=="L":
        #     return 2j*np.pi*freq*circuit.value()
        # elif circuit.key()=="E":
        #     return 1/(2j*np.pi*freq*circuit.value())
