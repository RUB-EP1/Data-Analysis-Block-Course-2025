// Check the browser forward pass (widgets/mnist-net.js) against Flux:
// logits for five test images at every checkpoint, from scripts/make_mnist_data.jl.
//   node slides/lectures/Lecture_3B/scripts/check_mnist_widget.mjs
import { readFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import vm from 'node:vm';

const here = new URL('.', import.meta.url).pathname;
const { mnistWeights, mnistForward } = createRequire(import.meta.url)('../widgets/mnist-net.js');
const ctx = { window: {} };
vm.runInNewContext(readFileSync(here + '../widgets/data_mnist.js', 'utf8'), ctx);
const D = ctx.window.DATA_MNIST;
const ref = JSON.parse(readFileSync(here + 'mnist_reference.json', 'utf8'));

let worst = 0;
D.checkpoints.forEach((cp, k) => {
  const W = mnistWeights(cp.w);
  ref.pixels.forEach((px, i) => {
    const { logits } = mnistForward(W, Float32Array.from(px));
    logits.forEach((v, j) => { worst = Math.max(worst, Math.abs(v - ref.logits[k][i][j])); });
  });
});
console.log(`largest |JS − Flux| logit difference: ${worst.toExponential(2)}`);
if (!(worst < 1e-3)) { console.error('mismatch'); process.exit(1); }
