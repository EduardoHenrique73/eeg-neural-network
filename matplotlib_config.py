
# Configurações de performance para Matplotlib
import matplotlib
matplotlib.use('Agg')  # Backend não interativo
matplotlib.rcParams['figure.max_open_warning'] = 0
matplotlib.rcParams['figure.dpi'] = 100
matplotlib.rcParams['savefig.dpi'] = 100
matplotlib.rcParams['savefig.bbox'] = 'tight'
matplotlib.rcParams['savefig.pad_inches'] = 0.1
matplotlib.rcParams['figure.figsize'] = [8, 6]
matplotlib.rcParams['font.size'] = 10

import matplotlib.pyplot as plt
plt.ioff()  # Desabilitar modo interativo

# Suprimir avisos
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
