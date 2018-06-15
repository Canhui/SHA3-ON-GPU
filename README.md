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
    <th class="tg-us36" rowspan="3">File Size (bytes)</th>
    <th class="tg-us36" colspan="4">Hash</th>
  </tr>
  <tr>
    <td class="tg-us36" colspan="2">Time (seconds)</td>
    <td class="tg-yw4l" colspan="2">Throughput (bytes per second)</td>
  </tr>
  <tr>
    <td class="tg-us36">CPU</td>
    <td class="tg-us36">GPU</td>
    <td class="tg-yw4l">CPU</td>
    <td class="tg-yw4l">GPU</td>
  </tr>
  <tr>
    <td class="tg-yw4l">1202</td>
    <td class="tg-yw4l">0.002656</td>
    <td class="tg-yw4l">0.000431</td>
    <td class="tg-yw4l">452560.24</td>
    <td class="tg-yw4l">2788863.11</td>
  </tr>
  <tr>
    <td class="tg-yw4l">4652</td>
    <td class="tg-yw4l">0.008600</td>
    <td class="tg-yw4l">0.000330</td>
    <td class="tg-yw4l">540930.23</td>
    <td class="tg-yw4l">14096969.69</td>
  </tr>
  <tr>
    <td class="tg-yw4l">9302</td>
    <td class="tg-yw4l">0.016400</td>
    <td class="tg-yw4l">0.000330</td>
    <td class="tg-yw4l">567195.12</td>
    <td class="tg-yw4l">28187878.78</td>
  </tr>
  <tr>
    <td class="tg-yw4l">18602</td>
    <td class="tg-yw4l">0.032475</td>
    <td class="tg-yw4l">0.000346</td>
    <td class="tg-yw4l">572809.85</td>
    <td class="tg-yw4l">53763005.78</td>
  </tr>
  <tr>
    <td class="tg-yw4l">37202</td>
    <td class="tg-yw4l">0.065230</td>
    <td class="tg-yw4l">0.000373</td>
    <td class="tg-yw4l">570320.40</td>
    <td class="tg-yw4l">99737265.42</td>
  </tr>
  <tr>
    <td class="tg-yw4l">74402</td>
    <td class="tg-yw4l">0.129495</td>
    <td class="tg-yw4l">0.000382</td>
    <td class="tg-yw4l">574555.00</td>
    <td class="tg-yw4l">194769633.41</td>
  </tr>
  <tr>
    <td class="tg-yw4l">148802</td>
    <td class="tg-yw4l">0.258204</td>
    <td class="tg-yw4l">0.000437</td>
    <td class="tg-yw4l">576296.26</td>
    <td class="tg-yw4l">340508009.15</td>
  </tr>
  <tr>
    <td class="tg-yw4l">297602</td>
    <td class="tg-yw4l">0.516079</td>
    <td class="tg-yw4l">0.000557</td>
    <td class="tg-yw4l">576659.77</td>
    <td class="tg-yw4l">534294434.47</td>
  </tr>
  <tr>
    <td class="tg-yw4l">595202</td>
    <td class="tg-yw4l">1.036339</td>
    <td class="tg-yw4l">0.000755</td>
    <td class="tg-yw4l">574331.37</td>
    <td class="tg-yw4l">788347019.87</td>
  </tr>
  <tr>
    <td class="tg-yw4l">1190402</td>
    <td class="tg-yw4l">2.060367</td>
    <td class="tg-yw4l">0.001204</td>
    <td class="tg-yw4l">577762.12</td>
    <td class="tg-yw4l">988705980.06</td>
  </tr>
</table>
