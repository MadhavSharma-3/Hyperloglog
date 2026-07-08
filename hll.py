import math
import random
import time


# --- SECTION 1: Hashing Utility (Checklist Item 1) ---
def murmur3_32(key, seed=0):
    """
    Pure Python implementation of MurmurHash3 (32-bit).
    Ensures consistent, well-distributed hashing without external dependencies.
    """
    key = bytearray(key.encode('utf-8')) if isinstance(key, str) else bytearray(key)
    length = len(key)
    n_blocks = length // 4
    
    h1 = seed
    c1 = 0xcc9e2d51
    c2 = 0x1b873593

    for i in range(n_blocks):
        k1 = key[i*4] | (key[i*4+1] << 8) | (key[i*4+2] << 16) | (key[i*4+3] << 24)
        k1 = (k1 * c1) & 0xffffffff
        k1 = ((k1 << 15) | (k1 >> 17)) & 0xffffffff
        k1 = (k1 * c2) & 0xffffffff
        h1 = (h1 ^ k1)
        h1 = ((h1 << 13) | (h1 >> 19)) & 0xffffffff
        h1 = (h1 * 5 + 0xe6546b64) & 0xffffffff

    tail_index = n_blocks * 4
    k1 = 0
    tail_len = length & 3
    if tail_len >= 3: k1 ^= key[tail_index + 2] << 16
    if tail_len >= 2: k1 ^= key[tail_index + 1] << 8
    if tail_len >= 1:
        k1 ^= key[tail_index]
        k1 = (k1 * c1) & 0xffffffff
        k1 = ((k1 << 15) | (k1 >> 17)) & 0xffffffff
        k1 = (k1 * c2) & 0xffffffff
        h1 ^= k1

    h1 ^= length
    h1 ^= (h1 >> 16)
    h1 = (h1 * 0x85ebca6b) & 0xffffffff
    h1 ^= (h1 >> 13)
    h1 = (h1 * 0xc2b2ae35) & 0xffffffff
    h1 ^= (h1 >> 16)
    return h1




# --- SECTION 2: The HyperLogLog Class (Checklist Items 2, 3, 4) ---
class HyperLogLog:
    def __init__(self, p=12):
        # Checklist Item 2a: Initialization
        if not (4 <= p <= 16):
            raise ValueError("p must be between 4 and 16")
        self.p = p
        self.m = 1 << p
        self.registers = [0] * self.m
        
        # Pre-calculate alpha constant
        if self.m == 16: self.alpha = 0.673
        elif self.m == 32: self.alpha = 0.697
        elif self.m == 64: self.alpha = 0.709
        else: self.alpha = 0.7213 / (1 + 1.079 / self.m)

    def _get_rho(self, w):
        # Helper to find rank (trailing zeros)
        if w == 0: return 32 - self.p + 1
        rho = 1
        while (w & 1) == 0:
            w >>= 1
            rho += 1
        return rho

    def add(self, item):
        # Checklist Item 2b: Bit Manipulation Logic
        x = murmur3_32(str(item))
        j = x >> (32 - self.p)                  # Register Index
        w = x & ((1 << (32 - self.p)) - 1)      # Remainder for Zero Counting
        rho = self._get_rho(w)
        
        if rho > self.registers[j]:
            self.registers[j] = rho 

    def count(self):
        # Checklist Item 3: Raw Estimate
        Z_inv = sum(2.0 ** -val for val in self.registers)
        E = self.alpha * (self.m ** 2) / Z_inv
        
        # Checklist Item 4: Range Corrections
        if E <= 2.5 * self.m: # Small Range (Linear Counting)
            V = self.registers.count(0)
            if V != 0:
                E = self.m * math.log(self.m / V)
        elif E > (1/30.0) * (1 << 32): # Large Range
            E = -(1 << 32) * math.log(1 - E / (1 << 32))
            
        return int(E)

    def merge(self, other):
        # Extra: Union Logic
        if self.p != other.p:
             raise ValueError("Precision mismatch")
        for i in range(self.m):
            if other.registers[i] > self.registers[i]:
                self.registers[i] = other.registers[i]




# --- SECTION 3: Analysis & Visualization (Checklist Item 5) ---
if __name__ == "__main__":

    print("--- HyperLogLog Project: Analysis Phase ---")
    
    # Setup Experiment
    p_val = 12
    hll = HyperLogLog(p=p_val)
    true_set = set()
    
    # Data points for plotting
    x_axis_actual = []
    y_axis_est = []
    y_axis_error = []
    
    # We will track 100,000 items
    total_items = 100000
    interval = max(1 , total_items//100)
    
    print(f"Running simulation with p={p_val} (m={2**p_val} registers)...")
    start_time = time.time()

    for i in range(1, total_items + 1):
        # Generate random data
        item = f"data_{random.random()}_{i}"
        
        hll.add(item)
        true_set.add(item)
        
        # Record data every interval items (to keep plot clean)
        if i % interval == 0:
            actual = len(true_set)
            est = hll.count()
            err = abs(est - actual) / actual * 100
            
            x_axis_actual.append(actual)
            y_axis_est.append(est)
            y_axis_error.append(err)
            
            if i % 10000 == 0:
                print(f"Step {i}: Actual={actual}, Est={est}, Error={err:.2f}%")

    print(f"Simulation done in {time.time() - start_time:.2f}s")

    # Plotting Logic
    try:
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot 1: Linearity
        ax1.plot(x_axis_actual, x_axis_actual, 'g--', label='Ground Truth', linewidth=1)
        ax1.plot(x_axis_actual, y_axis_est, 'b-', label='HLL Estimate', alpha=0.7)
        ax1.set_title('Linearity: Actual vs Estimate')
        ax1.set_xlabel('Actual Count')
        ax1.set_ylabel('HLL Count')
        ax1.legend()
        ax1.grid(True)
        
        # Plot 2: Error Rate
        ax2.plot(x_axis_actual, y_axis_error, 'r-', linewidth=1)
        ax2.set_title(f'Relative Error % (p={p_val})')
        ax2.set_xlabel('Cardinality')
        ax2.set_ylabel('Error %')
        # Add theoretical error line: 1.04 / sqrt(m)
        theoretical_err = 1.04 / math.sqrt(2**p_val) * 100
        ax2.axhline(y=theoretical_err, color='k', linestyle=':', label=f'Theoretical ({theoretical_err:.2f}%)')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
        print("Diagram generated.")
        
    except ImportError:
        print("\n[!] Matplotlib not installed. Install it to see the graphs: pip install matplotlib")
        print("Final Error: {:.2f}%".format(y_axis_error[-1]))


