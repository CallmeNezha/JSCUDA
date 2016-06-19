assert = require( "assert" )
colors = require( "colors" )
JC = require( "../linear" )


colors.setTheme({
  warmup: 'black'
  cpu: 'cyan'
  gpu: 'green'
  standardIEEE: [ 'white', 'underline' ]
})


# Test parameters
AbsoluteError = 1e-6
JC.cudaDeviceInit()


dumpMatrix = (mat) ->
  for r in [0...mat.numRow]
    oneRow = "["
    for c in [0...mat.numCol]
      oneRow += mat.elements[c*mat.numRow + r] + " "
    oneRow += "]"
    console.log( "#{ oneRow }" )



describe("MatrixD Functions Validation Test"
, ->
  @timeout( 0 ) # Disable mocha timeout for stress test

  it("matrix multiply scalar: ..."
  , () ->  #asynchronized test
    testLength = 1e3
    warmUpLength = 10

    # GPU pass
    for len in [testLength]
# Warm up the device
      v1h = new Float32Array( ( Math.random() for num in [0...len * len] ) )
      gpuResult = new Float32Array( len * len )

      matAd = new JC.MatrixD( len, len, v1h ) # Direct assign value to device memory
      scalar = Math.random()

      # Measure time
      start = Date.now()
      matAd.multiplyScalar( scalar )
      if len is warmUpLength
        tip = "Warm up pass "
        colorLog = "warmup"
      else
        tip = ""
        colorLog = "gpu"

      # Device -> Host
      matAd.copyTo( matAd.numRow * matAd.numCol, gpuResult )
      mid = Date.now()
      matAd.copyTo( matAd.numRow * matAd.numCol, gpuResult )
      end = Date.now()
      console.log( "\t#{tip}JC(GPU) <<< MatrixD::multiplyVector >>> #{ len }x#{ len } * #{ testLength } elements used:#{ Math.max( mid - start - ( end - mid ), 0 ) } ms"[colorLog] )
      matAd.destroy()

    # CPU pass
    start = Date.now()
    # Column major matrix
    for i in [0...len * len]
      v1h[i] *= scalar
    colorLog = "cpu"
    console.log( "\t#{tip}V8(CPU) <<< matrix multiply vector >>> #{ testLength }x#{ testLength } * #{ testLength } elements used:#{ Date.now() - start } ms"[colorLog] )


    # CPU & GPU comparison output
    meanError = 0
    for _, i in v1h
      meanError += v1h[i] - gpuResult[i]

    meanError /= v1h.length
    colorLog = "standardIEEE"
    console.log( "\tv8.CPU vs jc.GPU Mean absolute error: #{ meanError } , \<float32\> refer to IEEE-754"[colorLog] )

    # If failed
    assert( AbsoluteError > meanError > -AbsoluteError, "test failed" )
  )

  it("matrix multiply vector: ..."
  , () ->  #asynchronized test
    testLength = 1e3
    warmUpLength = 10

    # GPU pass
    for len in [testLength]
# Warm up the device
      v1h = new Float32Array( ( Math.random() for num in [0...len * len] ) )
      v2h = new Float32Array( ( Math.random() for num in [0...len] ) )
      gpuResult = new Float32Array( len )

      matAd = new JC.MatrixD( len, len, v1h ) # Direct assign value to device memory
      vBd = new JC.VectorD( v2h.length, v2h )
      vCd = new JC.VectorD( len )

      # Measure time
      start = Date.now()
      matAd.multiplyVector( vBd, vCd )
      if len is warmUpLength
        tip = "Warm up pass "
        colorLog = "warmup"
      else
        tip = ""
        colorLog = "gpu"

      # Device -> Host
      vCd.copyTo( vCd.length, gpuResult )
      mid = Date.now()
      vCd.copyTo( vCd.length, gpuResult )
      end = Date.now()
      console.log( "\t#{tip}JC(GPU) <<< MatrixD::multiplyVector >>> #{ len }x#{ len } * #{ testLength } elements used:#{ Math.max( mid - start - ( end - mid ), 0 ) } ms"[colorLog] )
      matAd.destroy()
      vBd.destroy()
      vCd.destroy()

    # CPU pass
    cpuresult = new Float32Array( gpuResult.length )
    start = Date.now()
    # Column major matrix
    for r in [0...len]
      element = 0
      for j in [0...len]
        v1e = v1h[j * len + r]
        v2e = v2h[j]
        element += v1e * v2e
      cpuresult[r] = element
    colorLog = "cpu"
    console.log( "\t#{tip}V8(CPU) <<< matrix multiply vector >>> #{ testLength }x#{ testLength } * #{ testLength } elements used:#{ Date.now() - start } ms"[colorLog] )


    # CPU & GPU comparison output
    meanError = 0
    for _, i in cpuresult
      meanError += cpuresult[i] - gpuResult[i]

    meanError /= cpuresult.length
    colorLog = "standardIEEE"
    console.log( "\tv8.CPU vs jc.GPU Mean absolute error: #{ meanError } , \<float32\> refer to IEEE-754"[colorLog] )

    # If failed
    assert( AbsoluteError > meanError > -AbsoluteError, "test failed" )
  )

  it("matrix's transpose multiply vector: ..."
  , () ->  #asynchronized test
    testLength = 1e3
    warmUpLength = 10

    # GPU pass
    for len in [testLength]
# Warm up the device
      v1h = new Float32Array( ( Math.random() for num in [0...len * len] ) )
      v2h = new Float32Array( ( Math.random() for num in [0...len] ) )
      gpuResult = new Float32Array( len )

      matAd = new JC.MatrixD( len, len, v1h ) # Direct assign value to device memory
      vBd = new JC.VectorD( v2h.length, v2h )
      vCd = new JC.VectorD( len )

      # Measure time
      start = Date.now()
      matAd.T().multiplyVector( vBd, vCd )
      if len is warmUpLength
        tip = "Warm up pass "
        colorLog = "warmup"
      else
        tip = ""
        colorLog = "gpu"

      # Device -> Host
      vCd.copyTo( vCd.length, gpuResult )
      mid = Date.now()
      vCd.copyTo( vCd.length, gpuResult )
      end = Date.now()
      console.log( "\t#{tip}JC(GPU) <<< MatrixD::T::multiplyVector >>> #{ len }x#{ len } * #{ testLength } elements used:#{ Math.max( mid - start - ( end - mid ), 0 ) } ms"[colorLog] )
      matAd.destroy()
      vBd.destroy()
      vCd.destroy()

    # CPU pass
    cpuresult = new Float32Array( gpuResult.length )
    start = Date.now()
    # Column major matrix
    for r in [0...len]
      element = 0
      for j in [0...len]
        v1e = v1h[r * len + j]
        v2e = v2h[j]
        element += v1e * v2e
      cpuresult[r] = element
    colorLog = "cpu"
    console.log( "\t#{tip}V8(CPU) <<< matrix's transpose multiply vector >>> #{ testLength }x#{ testLength } * #{ testLength } elements used:#{ Date.now() - start } ms"[colorLog] )


    # CPU & GPU comparison output
    meanError = 0
    for _, i in cpuresult
      meanError += cpuresult[i] - gpuResult[i]

    meanError /= cpuresult.length
    colorLog = "standardIEEE"
    console.log( "\tv8.CPU vs jc.GPU Mean absolute error: #{ meanError } , \<float32\> refer to IEEE-754"[colorLog] )

    # If failed
    assert( AbsoluteError > meanError > -AbsoluteError, "test failed" )
  )

  it("matrix multiplication: ..."
  , () ->  #asynchronized test
    testLength = 1e3
    warmUpLength = 10

    # GPU pass
    for len in [testLength]
# Warm up the device
      v1h = new Float32Array( ( Math.random() for num in [0...len * len] ) )
      v2h = new Float32Array( ( Math.random() for num in [0...len * len] ) )
      gpuResult = new Float32Array( len * len )

      matAd = new JC.MatrixD( len, len, v1h ) # Direct assign value to device memory
      matBd = new JC.MatrixD( len, len, v2h )
      matCd = new JC.MatrixD( len, len )

      # Measure time
      start = Date.now()
      matAd.multiplyMatrix(matBd, matCd)
      if len is warmUpLength
        tip = "Warm up pass "
        colorLog = "warmup"
      else
        tip = ""
        colorLog = "gpu"

      # Device -> Host
      matCd.copyTo( matCd.numCol * matCd.numRow, gpuResult  )
      mid = Date.now()
      matCd.copyTo( matCd.numCol * matCd.numRow, gpuResult  )
      end = Date.now()
      console.log( "\t#{tip}JC(GPU) <<< MatrixD::multiplyMatrix >>> #{ len }x#{ len } * #{ len }x#{ len } elements used:#{ Math.max( mid - start - ( end - mid ), 0 ) } ms"[colorLog] )
      matAd.destroy()
      matBd.destroy()
      matCd.destroy()

    # CPU pass
    cpuresult = new Float32Array( gpuResult.length )
    start = Date.now()
    # Column major matrix
    for r in [0...len]
      for c in [0...len]
        element = 0
        for j in [0...len]
          v1e = v1h[j * len + r]
          v2e = v2h[c * len + j]
          element += v1e * v2e
        cpuresult[c * len + r] = element
    colorLog = "cpu"
    console.log( "\t#{tip}V8(CPU) <<< matrix multiplication >>> #{ testLength }x#{ testLength } * #{ testLength }x#{ testLength } elements used:#{ Date.now() - start } ms"[colorLog] )


    # CPU & GPU comparison output
    meanError = 0
    for _, i in cpuresult
      meanError += cpuresult[i] - gpuResult[i]

    meanError /= cpuresult.length
    colorLog = "standardIEEE"
    console.log( "\tv8.CPU vs jc.GPU Mean absolute error: #{ meanError } , \<float32\> refer to IEEE-754"[colorLog] )

    # If failed
    assert( AbsoluteError > meanError > -AbsoluteError, "test failed" )
  )

  it("matrices' transposes' multiplication: ..."
  , () ->  #asynchronized test
    testLength = 1e3
    warmUpLength = 10

    # GPU pass
    for len in [testLength]
# Warm up the device
      v1h = new Float32Array( ( Math.random() for num in [0...len * len] ) )
      v2h = new Float32Array( ( Math.random() for num in [0...len * len] ) )
      gpuResult = new Float32Array( len * len )

      matAd = new JC.MatrixD( len, len, v1h ) # Direct assign value to device memory
      matBd = new JC.MatrixD( len, len, v2h )
      matCd = new JC.MatrixD( len, len )

      # Measure time
      start = Date.now()
      matAd.T().multiplyMatrix(matBd.T(), matCd)
      if len is warmUpLength
        tip = "Warm up pass "
        colorLog = "warmup"
      else
        tip = ""
        colorLog = "gpu"

      # Device -> Host
      matCd.copyTo( matCd.numCol * matCd.numRow, gpuResult  )
      mid = Date.now()
      matCd.copyTo( matCd.numCol * matCd.numRow, gpuResult  )
      end = Date.now()
      console.log( "\t#{tip}JC(GPU) <<< MatrixD::T::multiplyMatrix >>> #{ len }x#{ len } * #{ len }x#{ len } elements used:#{ Math.max( mid - start - ( end - mid ), 0 ) } ms"[colorLog] )
      matAd.destroy()
      matBd.destroy()
      matCd.destroy()

    # CPU pass
    cpuresult = new Float32Array( gpuResult.length )
    start = Date.now()
    # Column major matrix
    for r in [0...len]
      for c in [0...len]
        element = 0
        for j in [0...len]
          v1e = v1h[r * len + j]
          v2e = v2h[j * len + c]
          element += v1e * v2e
        cpuresult[c * len + r] = element
    colorLog = "cpu"
    console.log( "\t#{tip}V8(CPU) <<< matrices' transposes' multiplication >>> #{ testLength }x#{ testLength } * #{ testLength }x#{ testLength } elements used:#{ Date.now() - start } ms"[colorLog] )


    # CPU & GPU comparison output
    meanError = 0
    for _, i in cpuresult
      meanError += cpuresult[i] - gpuResult[i]

    meanError /= cpuresult.length
    colorLog = "standardIEEE"
    console.log( "\tv8.CPU vs jc.GPU Mean absolute error: #{ meanError } , \<float32\> refer to IEEE-754"[colorLog] )

    # If failed
    assert( AbsoluteError > meanError > -AbsoluteError, "test failed" )
  )

  it("matrices'batch multiplication: ..."
  , () ->  #asynchronized test

    batchA = ( new JC.MatrixD( 2, 2, new Float32Array( (num for num in [0...4]) ) ) for i in [0...1] )
    batchB = ( new JC.MatrixD( 2, 2, new Float32Array( (num for num in [0...4]) ) ) for i in [0...1] )
    batchC = ( new JC.MatrixD( 2, 2, new Float32Array( (num for num in [0...4]) ) ) for i in [0...1] )

    matrixWatch = batchC[0]
    hostOut = new Float32Array( 4 )
    matrixWatch.copyTo( 4, hostOut )
    console.log( "Out-put before batch multiply: #{ hostOut }" )

    mbdA = new JC.MatrixBatchD( 2, 2, batchA )
    mbdB = new JC.MatrixBatchD( 2, 2, batchB )
    mbdC = new JC.MatrixBatchD( 2, 2, batchC )

    mbdA.T().multiplyMatrixBatch( mbdB, mbdC )

    matrixWatch.copyTo( 4, hostOut )
    console.log( "Out-put before batch multiply: #{ hostOut }" )

    mbdA.destroy()
    mbdB.destroy()
    mbdC.destroy()
  )
)
