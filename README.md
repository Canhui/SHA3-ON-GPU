# 0. Environment Specification
```
OS: Ubuntu 16.04 LTS, 64-bit
Memory: 11.5 GiB
CPU: Intel Core i5-7200U @ 2.50GHz × 4
GPU: GeForce 940MX/PCIe/SSE2
```

# 1. SHA3-ON-GPU
Implement SHA3 on CPU & GPU

# 2. Usage
```
gcc cpu.c -o cpu
```

```
nvcc gpu.cu -o gpu
```

# 3. Experimental Results
<table class="tg">
  <tr>
    <th class="tg-c3ow" rowspan="3">File Size (KBytes)</th>
    <th class="tg-c3ow" colspan="4">Hash</th>
    <th class="tg-c3ow" rowspan="3">Speedup</th>
  </tr>
  <tr>
    <td class="tg-us36" colspan="2">Time (seconds)</td>
    <td class="tg-us36" colspan="2">Throughput (KBytes per second)</td>
  </tr>
  <tr>
    <td class="tg-us36">CPU</td>
    <td class="tg-us36">GPU</td>
    <td class="tg-us36">CPU</td>
    <td class="tg-us36">GPU</td>
  </tr>
  <tr>
    <td class="tg-us36">80</td>
    <td class="tg-us36">0.059094</td>
    <td class="tg-us36">0.05876</td>
    <td class="tg-us36">1128</td>
    <td class="tg-us36">1360</td>
    <td class="tg-us36">1.2</td>
  </tr>
  <tr>
    <td class="tg-us36">200</td>
    <td class="tg-us36">0.179942</td>
    <td class="tg-us36">0.059780</td>
    <td class="tg-us36">1111</td>
    <td class="tg-us36">3345</td>
    <td class="tg-us36">3.0</td>
  </tr>
  <tr>
    <td class="tg-us36">400</td>
    <td class="tg-us36">0.355316</td>
    <td class="tg-us36">0.045508</td>
    <td class="tg-us36">1125</td>
    <td class="tg-us36">8789</td>
    <td class="tg-us36">7.8</td>
  </tr>
  <tr>
    <td class="tg-us36">800</td>
    <td class="tg-us36">0.718017</td>
    <td class="tg-us36">0.059382</td>
    <td class="tg-us36">1114</td>
    <td class="tg-us36">13472</td>
    <td class="tg-us36">12</td>
  </tr>
  <tr>
    <td class="tg-us36">1024</td>
    <td class="tg-us36">0.905427</td>
    <td class="tg-us36">0.059193</td>
    <td class="tg-us36">1130</td>
    <td class="tg-us36">17299</td>
    <td class="tg-us36">15</td>
  </tr>
</table>
