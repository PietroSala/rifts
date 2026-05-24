# Third-party notices

This repository vendors source code from third-party projects. Each entry
below records the project, where its source lives in this tree, the
license it ships under, and a verbatim copy of (or pointer to) the
upstream copyright notice.

## RFxpl

* Upstream: <https://github.com/izzayacine/RFxpl>
* Author: Yacine Izza (2023)
* Vendored at: `third_party/RFxpl/`
* License: MIT
* What we use: RFxpl is the reference implementation of the line of work
  that defines the Max-iAXp baseline; it is the baseline against which
  `scripts/max_iaxp/*.py` runs.
* Upstream-requested citations (please cite if you publish work that
  builds on RFxpl):

  > Izza, Y., Marques-Silva, J. (2021). *On Explaining Random Forests
  > with SAT*. IJCAI 2021, pages 2584–2591.
  > [doi:10.24963/ijcai.2021/356](https://doi.org/10.24963/ijcai.2021/356)

  > Izza, Y., Ignatiev, A., Stuckey, P.\,J., Marques-Silva, J. (2024).
  > *Delivering Inflated Explanations*. AAAI 2024, pages 12744–12753.
  > [doi:10.1609/aaai.v38i11.29170](https://doi.org/10.1609/aaai.v38i11.29170)

The unmodified MIT license text from upstream is preserved at
`third_party/RFxpl/LICENSE` and is reproduced below for convenience:

```
MIT License

Copyright (c) 2023 Yacine Izza

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### Modifications from upstream

The vendored copy is upstream RFxpl with one omission: the 14 MB
`resources.tar.gz` bundle of test datasets is not shipped, because it is
not exercised by our Max-iAXp wrapper. Re-fetching that archive from
upstream restores it.
