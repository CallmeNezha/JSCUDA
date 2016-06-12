JC = require("./../jc/build/Debug/jc.node")
JC.cudaDeviceInit()
f32aa = (new Float32Array(num) for num in [0...100])
array = (new JC.DeviceFloat32Array(f32a) for f32a in f32aa)
console.log(array.length)
for a, i in array
  array[i] = null

for i in [0...100]
  array.push(i)

console.log(array.length)
JC.cudaDeviceReset()
