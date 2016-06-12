assert = require("assert")
JC = require("./../jc/build/Debug/jc.node")


dumpMatrix = (mat) ->
  for r in [0...mat.numRow]
    oneRow = "["
    for c in [0...mat.numCol]
      oneRow += mat.elements[c*mat.numRow + r] + " "
    oneRow += "]"
    console.log("#{oneRow}")


describe("CUDA Base Utils Validation Check"
, ->
  it("CUDA: cudaInitDevice"
  , (done) ->  #asynchronized test
    assert.equal(typeof JC.cudaDeviceInit, 'function')
    done()
  )
  it("CUDA: cudaDeviceReset"
  , (done) ->  #asynchronized test
    assert.equal(typeof JC.cudaDeviceReset, 'function')
    done()
  )
)

describe("CUDA Functions Validation Test"
, ->
  JC.cudaDeviceInit()

  it("CUDA: matrixMulMatrix(m1,m2,m3) Warmup..."
  , () ->  #asynchronized test
    assert.equal(typeof JC.vectorAdd, 'function')
    v1 = new Float32Array((num for num in [0...6]))
    v2 = new Float32Array((num for num in [0...15]))
    v3 = new Float32Array(10)
    mat1 = {
      numRow: 2
      numCol: 3
      elements: v1
    }
    mat2 = {
      numRow: 3
      numCol: 5
      elements: v2
    }
    mat3 = {
      numRow: 2
      numCol: 5
      elements: v3
    }
    start = Date.now()
    JC.matrixMulMatrix(mat1, mat2, mat3)
    end = Date.now()
    console.log("Matrix2x3 #1}")
    console.log("Matrix3x5 #2}")
    console.log("Matrix2x5 #3}")
    console.log("\t elapse time:  #{ end - start } ms")
    console.log("------------ * ------------------------- * -------------")
  )
  it("CUDA: matrixMulMatrix(m1,m2,m3) test #1"
  , () ->  #asynchronized test
    assert.equal(typeof JC.vectorAdd, 'function')
    v1 = new Float32Array((num for num in [0...6]))
    v2 = new Float32Array((num for num in [0...15]))
    v3 = new Float32Array(10)
    mat1 = {
      numRow: 2
      numCol: 3
      elements: v1
    }
    mat2 = {
      numRow: 3
      numCol: 5
      elements: v2
    }
    mat3 = {
      numRow: 2
      numCol: 5
      elements: v3
    }
    start = Date.now()
    JC.matrixMulMatrix(mat1, mat2, mat3)
    end = Date.now()
    console.log("------------ * ------------------------- * -------------")
    console.log("Matrix2x3 #1:}")
    dumpMatrix(mat1)
    console.log("")
    console.log("Matrix3x5 #2:}")
    dumpMatrix(mat2)
    console.log("")
    console.log("Matrix2x5 #3:}")
    dumpMatrix(mat3)
    console.log("")

    console.log("\t elapse time:  #{ end - start } ms")
    console.log("------------ * ------------------------- * -------------")
  )
  it("CUDA: matrixMulMatrix(m1,m2,m3) test #1"
  , () ->  #asynchronized test
    assert.equal(typeof JC.vectorAdd, 'function')
    v1 = new Float32Array((Math.random() for num in [0...10000]))
    v2 = new Float32Array((Math.random() for num in [0...10000]))
    v3 = new Float32Array(10000)
    mat1 = {
      numRow: 100
      numCol: 100
      elements: v1
    }
    mat2 = {
      numRow: 100
      numCol: 100
      elements: v2
    }
    mat3 = {
      numRow: 100
      numCol: 100
      elements: v3
    }
    start = Date.now()
    JC.matrixMulMatrix(mat1, mat2, mat3)
    end = Date.now()
    console.log("------------ * ------------------------- * -------------")
    console.log("Matrix 100 x 100 #1}")
    console.log("Matrix 100 x 100 #2}")
    console.log("Matrix 100 x 100 #3}")
    console.log("\t elapse time:  #{ end - start } ms")
    console.log("------------ * ------------------------- * -------------")
  )

  it("CUDA: vectorAdd(v1,v2,v3) Warmup..."
  , () ->  #asynchronized test
    assert.equal(typeof JC.vectorAdd, 'function')
    v1 = new Float32Array((Math.random() for num in [0...100]))
    v2 = new Float32Array((Math.random() for num in [0...100]))
    v3 = new Float32Array(100)
    start = Date.now()
    JC.vectorAdd(v1, v2, v3)
    end = Date.now()
    console.log("------------ * ------------------------- * -------------")
    console.log("\t elapse time:  #{ end - start } ms")
    console.log("------------ * ------------------------- * -------------")
  )

  it("CUDA: vectorAdd(v1,v2,v3) test #1"
  , () ->  #asynchronized test
    assert.equal(typeof JC.vectorAdd, 'function')
    v1 = new Float32Array((Math.random() for num in [0...1e6]))
    v2 = new Float32Array((Math.random() for num in [0...1e6]))
    v3 = new Float32Array(1e6)
    console.log("------------ * ------------------------- * -------------")
    start = Date.now()
    JC.vectorAdd(v1, v2, v3)
    end = Date.now()
    console.log("\t Array #1:#{v1[0...3]}... 1 million elements")
    console.log("\t Array #2:#{v2[0...3]}... 1 million elements")
    console.log("\t Array Out:#{v3[0...3]}... 1 million elements")
    console.log("\t elapse time:  #{ end - start } ms")
    console.log("------------ * ------------------------- * -------------")
  )
  it("CUDA: vectorAdd(v1,v2,v3) test #2"
  , () ->  #asynchronized test
    assert.equal(typeof JC.vectorAdd, 'function')
    v1 = new Float32Array((Math.random() for num in [0...100]))
    v2 = new Float32Array((Math.random() for num in [0...100]))
    v3 = new Float32Array(100)
    start = Date.now()
    JC.vectorAdd(v1, v2, v3)
    end = Date.now()
    console.log("------------ * ------------------------- * -------------")
    console.log("\t Array #1:#{v1[0...3]}... 10 elements")
    console.log("\t Array #2:#{v2[0...3]}... 10 elements")
    console.log("\t Array Out:#{v3[0...3]}... 10 elements")
    console.log("\t elapse time:  #{ end - start } ms")
    console.log("------------ * ------------------------- * -------------")
  )
  JC.cudaDeviceReset()
)



describe("Linear Library Completeness"
, ->
  lb = require("./../linear")
  it("N dimensions Vector inner product"
  , (done) ->  #asynchronized test
      Vector = lb?.Vector
      v1 = new Vector(5,[1,2,3,4,5])
      v2 = new Vector(5,[1,2,3,4,5])
      assert.equal(v1.innerProduct(v2), 55)
      done()
  )
#  it("CUDA test: CUDA functions check"
#  , (done) ->  #asynchronized test
#      assert.equal(typeof JC.cudaDeviceInit, 'function')
#    done()
#  )
)