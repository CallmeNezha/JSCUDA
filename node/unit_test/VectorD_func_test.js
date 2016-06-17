// Generated by CoffeeScript 1.10.0
(function() {
  var AbsoluteError, JC, assert, colors, dumpMatrix;

  assert = require("assert");

  colors = require("colors");

  JC = require("../linear");

  colors.setTheme({
    warmup: 'black',
    cpu: 'cyan',
    gpu: 'green',
    standardIEEE: ['white', 'underline']
  });

  AbsoluteError = 1e-6;

  JC.cudaDeviceInit();

  dumpMatrix = function(mat) {
    var c, k, l, oneRow, r, ref, ref1, results;
    results = [];
    for (r = k = 0, ref = mat.numRow; 0 <= ref ? k < ref : k > ref; r = 0 <= ref ? ++k : --k) {
      oneRow = "[";
      for (c = l = 0, ref1 = mat.numCol; 0 <= ref1 ? l < ref1 : l > ref1; c = 0 <= ref1 ? ++l : --l) {
        oneRow += mat.elements[c * mat.numRow + r] + " ";
      }
      oneRow += "]";
      results.push(console.log("" + oneRow));
    }
    return results;
  };

  describe("CUDA Base Utils Validation Check", function() {
    this.timeout(0);
    it("CUDA: cudaInitDevice", function(done) {
      assert.equal(typeof JC.cudaDeviceInit, 'function');
      return done();
    });
    return it("CUDA: cudaDeviceReset", function(done) {
      assert.equal(typeof JC.cudaDeviceReset, 'function');
      return done();
    });
  });

  describe("VectorD Functions Validation Test", function() {
    it("vector add: ...", function() {
      var colorLog, e, end, gpuResult, i, k, l, len, len1, len2, len3, m, meanError, mid, num, ref, start, testLength, tip, v1h, v2h, vAd, vBd, warmUpLength;
      testLength = 1e7;
      warmUpLength = 10;
      ref = [warmUpLength, testLength];
      for (k = 0, len1 = ref.length; k < len1; k++) {
        len = ref[k];
        v1h = new Float32Array((function() {
          var l, ref1, results;
          results = [];
          for (num = l = 0, ref1 = len; 0 <= ref1 ? l < ref1 : l > ref1; num = 0 <= ref1 ? ++l : --l) {
            results.push(Math.random());
          }
          return results;
        })());
        v2h = new Float32Array((function() {
          var l, ref1, results;
          results = [];
          for (num = l = 0, ref1 = len; 0 <= ref1 ? l < ref1 : l > ref1; num = 0 <= ref1 ? ++l : --l) {
            results.push(Math.random());
          }
          return results;
        })());
        gpuResult = new Float32Array(len);
        vAd = new JC.VectorD(v1h.length, v1h);
        vBd = new JC.VectorD(len);
        vBd.copyFrom(len, v2h);
        start = Date.now();
        vAd.add(vBd);
        if (len === warmUpLength) {
          tip = "Warm up pass ";
          colorLog = "warmup";
        } else {
          tip = "";
          colorLog = "gpu";
        }
        vAd.copyTo(len, gpuResult);
        mid = Date.now();
        vAd.copyTo(len, gpuResult);
        end = Date.now();
        console.log(("\t" + tip + "JC(GPU) <<< VectorD::add >>> " + len + " elements used:" + (Math.max(mid - start - (end - mid), 0)) + " ms")[colorLog]);
        vAd.destroy();
        vBd.destroy();
      }
      colorLog = "cpu";
      start = Date.now();
      for (i = l = 0, len2 = v2h.length; l < len2; i = ++l) {
        e = v2h[i];
        v1h[i] += e;
      }
      console.log(("\tV8(CPU) <<< vector add >>> " + testLength + " elements used:" + (Date.now() - start) + " ms")[colorLog]);
      meanError = 0;
      for (i = m = 0, len3 = v1h.length; m < len3; i = ++m) {
        e = v1h[i];
        meanError += Math.abs(e - gpuResult[i]);
      }
      meanError /= v1h.length;
      colorLog = "standardIEEE";
      console.log(("\tv8.CPU vs jc.GPU Mean absolute error: " + meanError + " , \<float32\> refer to IEEE-754")[colorLog]);
      return assert((AbsoluteError > meanError && meanError > -AbsoluteError), "test failed");
    });
    it("vector copy: ...", function() {
      var colorLog, e, end, gpuResult, i, k, l, len, len1, len2, len3, m, meanError, mid, num, ref, start, testLength, tip, v1h, v2h, vAd, vBd, warmUpLength;
      testLength = 1e7;
      warmUpLength = 10;
      ref = [testLength];
      for (k = 0, len1 = ref.length; k < len1; k++) {
        len = ref[k];
        v1h = new Float32Array((function() {
          var l, ref1, results;
          results = [];
          for (num = l = 0, ref1 = len; 0 <= ref1 ? l < ref1 : l > ref1; num = 0 <= ref1 ? ++l : --l) {
            results.push(Math.random());
          }
          return results;
        })());
        v2h = new Float32Array((function() {
          var l, ref1, results;
          results = [];
          for (num = l = 0, ref1 = len; 0 <= ref1 ? l < ref1 : l > ref1; num = 0 <= ref1 ? ++l : --l) {
            results.push(Math.random());
          }
          return results;
        })());
        gpuResult = new Float32Array(len);
        vAd = new JC.VectorD(v1h.length, v1h);
        vBd = new JC.VectorD(len);
        vBd.copyFrom(len, v2h);
        start = Date.now();
        vAd.add(vBd);
        if (len === warmUpLength) {
          tip = "Warm up pass ";
          colorLog = "warmup";
        } else {
          tip = "";
          colorLog = "gpu";
        }
        vAd.copyTo(len, gpuResult);
        mid = Date.now();
        vAd.copyTo(len, gpuResult);
        end = Date.now();
        console.log(("\t" + tip + "JC(GPU) <<< VectorD::copy >>> " + len + " elements used:" + (Math.max(mid - start - (end - mid), 0)) + " ms")[colorLog]);
        vAd.destroy();
        vBd.destroy();
      }
      start = Date.now();
      for (i = l = 0, len2 = v2h.length; l < len2; i = ++l) {
        e = v2h[i];
        v1h[i] += e;
      }
      colorLog = "cpu";
      console.log(("\t" + tip + "V8(CPU) <<< vector copy >>> " + testLength + " elements used:" + (Date.now() - start) + " ms")[colorLog]);
      meanError = 0;
      for (i = m = 0, len3 = v1h.length; m < len3; i = ++m) {
        e = v1h[i];
        meanError += Math.abs(e - gpuResult[i]);
      }
      meanError /= v1h.length;
      colorLog = "standardIEEE";
      console.log(("\tv8.CPU vs jc.GPU Mean absolute error: " + meanError + " , \<float32\> refer to IEEE-754")[colorLog]);
      return assert((AbsoluteError > meanError && meanError > -AbsoluteError), "test failed");
    });
    it("vector multiply scalar: ...", function() {
      var _, colorLog, e, end, gpuResult, i, k, l, len, len1, len2, len3, m, meanError, mid, num, ref, scalar, start, testLength, tip, v1h, vAd, warmUpLength;
      testLength = 1e7;
      warmUpLength = 10;
      scalar = new Float32Array([Math.random()]);
      ref = [testLength];
      for (k = 0, len1 = ref.length; k < len1; k++) {
        len = ref[k];
        v1h = new Float32Array((function() {
          var l, ref1, results;
          results = [];
          for (num = l = 0, ref1 = len; 0 <= ref1 ? l < ref1 : l > ref1; num = 0 <= ref1 ? ++l : --l) {
            results.push(Math.random());
          }
          return results;
        })());
        gpuResult = new Float32Array(len);
        vAd = new JC.VectorD(v1h.length, v1h);
        start = Date.now();
        vAd.multiplyScalar(scalar[0]);
        if (len === warmUpLength) {
          tip = "Warm up pass ";
          colorLog = "warmup";
        } else {
          tip = "";
          colorLog = "gpu";
        }
        vAd.copyTo(len, gpuResult);
        mid = Date.now();
        vAd.copyTo(len, gpuResult);
        end = Date.now();
        console.log(("\t" + tip + "JC(GPU) <<< VectorD::multiplyScalar >>> " + len + " elements used:" + (Math.max(mid - start - (end - mid), 0)) + " ms")[colorLog]);
        vAd.destroy();
      }
      start = Date.now();
      for (i = l = 0, len2 = v1h.length; l < len2; i = ++l) {
        _ = v1h[i];
        v1h[i] *= scalar[0];
      }
      colorLog = "cpu";
      console.log(("\t" + tip + "V8(CPU) <<< vector multiply scalar >>> " + testLength + " elements used:" + (Date.now() - start) + " ms")[colorLog]);
      meanError = 0;
      for (i = m = 0, len3 = v1h.length; m < len3; i = ++m) {
        e = v1h[i];
        meanError += Math.abs(e - gpuResult[i]);
      }
      meanError /= v1h.length;
      colorLog = "standardIEEE";
      console.log(("\tv8.CPU vs jc.GPU Mean absolute error: " + meanError + " , \<float32\> refer to IEEE-754")[colorLog]);
      return assert((AbsoluteError > meanError && meanError > -AbsoluteError), "test failed");
    });
    it("vector dot: ...", function() {
      var colorLog, cpuresult, gpuResult, i, k, l, len, len1, meanError, num, ref, ref1, start, testLength, tip, v1h, v2h, vAd, vBd, warmUpLength;
      testLength = 1e7;
      warmUpLength = 10;
      ref = [testLength];
      for (k = 0, len1 = ref.length; k < len1; k++) {
        len = ref[k];
        v1h = new Float32Array((function() {
          var l, ref1, results;
          results = [];
          for (num = l = 0, ref1 = len; 0 <= ref1 ? l < ref1 : l > ref1; num = 0 <= ref1 ? ++l : --l) {
            results.push(Math.random());
          }
          return results;
        })());
        v2h = new Float32Array((function() {
          var l, ref1, results;
          results = [];
          for (num = l = 0, ref1 = len; 0 <= ref1 ? l < ref1 : l > ref1; num = 0 <= ref1 ? ++l : --l) {
            results.push(Math.random());
          }
          return results;
        })());
        vAd = new JC.VectorD(v1h.length, v1h);
        vBd = new JC.VectorD(len);
        vBd.copyFrom(len, v2h);
        start = Date.now();
        gpuResult = vAd.dot(vBd);
        if (len === warmUpLength) {
          tip = "Warm up pass ";
          colorLog = "warmup";
        } else {
          tip = "";
          colorLog = "gpu";
        }
        console.log(("\t" + tip + "JC(GPU) <<< VectorD::dot >>> " + len + " elements used:" + (Date.now() - start) + " ms")[colorLog]);
        vAd.destroy();
        vBd.destroy();
      }
      start = Date.now();
      cpuresult = 0;
      for (i = l = 0, ref1 = v1h.length; 0 <= ref1 ? l < ref1 : l > ref1; i = 0 <= ref1 ? ++l : --l) {
        cpuresult += v1h[i] * v2h[i];
      }
      colorLog = "cpu";
      console.log(("\t" + tip + "V8(CPU) <<< vector dot >>> " + testLength + " elements used:" + (Date.now() - start) + " ms")[colorLog]);
      meanError = (cpuresult - gpuResult) / v1h.length;
      colorLog = "standardIEEE";
      console.log(("\tv8.CPU vs jc.GPU Mean absolute error: " + meanError + " , \<float32\> refer to IEEE-754")[colorLog]);
      return assert((AbsoluteError > meanError && meanError > -AbsoluteError), "test failed");
    });
    it("vector Euclidean norm: ...", function() {
      var colorLog, cpuresult, e, gpuResult, k, l, len, len1, len2, meanError, num, ref, start, testLength, tip, v1h, vAd, warmUpLength;
      testLength = 1e7;
      warmUpLength = 10;
      ref = [testLength];
      for (k = 0, len1 = ref.length; k < len1; k++) {
        len = ref[k];
        v1h = new Float32Array((function() {
          var l, ref1, results;
          results = [];
          for (num = l = 0, ref1 = len; 0 <= ref1 ? l < ref1 : l > ref1; num = 0 <= ref1 ? ++l : --l) {
            results.push(Math.random());
          }
          return results;
        })());
        vAd = new JC.VectorD(v1h.length, v1h);
        start = Date.now();
        gpuResult = vAd.norm();
        if (len === warmUpLength) {
          tip = "Warm up pass ";
          colorLog = "warmup";
        } else {
          tip = "";
          colorLog = "gpu";
        }
        console.log(("\t" + tip + "JC(GPU) <<< VectorD::norm >>> " + len + " elements used:" + (Date.now() - start) + " ms")[colorLog]);
        vAd.destroy();
      }
      start = Date.now();
      cpuresult = 0;
      for (l = 0, len2 = v1h.length; l < len2; l++) {
        e = v1h[l];
        cpuresult += e * e;
      }
      cpuresult = Math.sqrt(cpuresult);
      colorLog = "cpu";
      console.log(("\t" + tip + "V8(CPU) <<< vector norm >>> " + testLength + " elements used:" + (Date.now() - start) + " ms")[colorLog]);
      meanError = (cpuresult - gpuResult) / v1h.length;
      colorLog = "standardIEEE";
      console.log(("\tv8.CPU vs jc.GPU Mean absolute error: " + meanError + " , \<float32\> refer to IEEE-754")[colorLog]);
      return assert((AbsoluteError > meanError && meanError > -AbsoluteError), "test failed");
    });
    return it("vector tensor product: ...", function() {
      var _, colorLog, cpuresult, end, gpuResult, i, j, k, l, len, len1, len2, len3, len4, m, matd, meanError, mid, n, num, ref, start, testLength, tip, v1e, v1h, v2e, v2h, vAd, vBd, warmUpLength;
      testLength = 1e3;
      warmUpLength = 10;
      ref = [testLength];
      for (k = 0, len1 = ref.length; k < len1; k++) {
        len = ref[k];
        v1h = new Float32Array((function() {
          var l, ref1, results;
          results = [];
          for (num = l = 0, ref1 = len; 0 <= ref1 ? l < ref1 : l > ref1; num = 0 <= ref1 ? ++l : --l) {
            results.push(Math.random());
          }
          return results;
        })());
        v2h = new Float32Array((function() {
          var l, ref1, results;
          results = [];
          for (num = l = 0, ref1 = len; 0 <= ref1 ? l < ref1 : l > ref1; num = 0 <= ref1 ? ++l : --l) {
            results.push(Math.random());
          }
          return results;
        })());
        gpuResult = new Float32Array(v1h.length * v2h.length);
        vAd = new JC.VectorD(v1h.length, v1h);
        vBd = new JC.VectorD(v2h.length, v2h);
        matd = new JC.MatrixD(v1h.length, v2h.length);
        start = Date.now();
        vAd.tensor(vBd, matd);
        if (len === warmUpLength) {
          tip = "Warm up pass ";
          colorLog = "warmup";
        } else {
          tip = "";
          colorLog = "gpu";
        }
        matd.copyTo(matd.numCol * matd.numRow, gpuResult);
        mid = Date.now();
        matd.copyTo(matd.numCol * matd.numRow, gpuResult);
        end = Date.now();
        console.log(("\t" + tip + "JC(GPU) <<< VectorD::tensor >>> " + len + "x" + len + " elements used:" + (Math.max(mid - start - (end - mid), 0)) + " ms")[colorLog]);
        matd.destroy();
        vAd.destroy();
        vBd.destroy();
      }
      cpuresult = new Float32Array(v1h.length * v2h.length);
      start = Date.now();
      for (j = l = 0, len2 = v2h.length; l < len2; j = ++l) {
        v2e = v2h[j];
        for (i = m = 0, len3 = v1h.length; m < len3; i = ++m) {
          v1e = v1h[i];
          cpuresult[v1h.length * j + i] = v2e * v1e;
        }
      }
      colorLog = "cpu";
      console.log(("\t" + tip + "V8(CPU) <<< vector tensor product >>> " + testLength + "x" + testLength + " elements used:" + (Date.now() - start) + " ms")[colorLog]);
      meanError = 0;
      for (i = n = 0, len4 = cpuresult.length; n < len4; i = ++n) {
        _ = cpuresult[i];
        meanError += cpuresult[i] - gpuResult[i];
      }
      meanError /= cpuresult.length;
      colorLog = "standardIEEE";
      console.log(("\tv8.CPU vs jc.GPU Mean absolute error: " + meanError + " , \<float32\> refer to IEEE-754")[colorLog]);
      return assert((AbsoluteError > meanError && meanError > -AbsoluteError), "test failed");
    });
  });

}).call(this);

//# sourceMappingURL=VectorD_func_test.js.map
