# LTS-inspired mapping

Primary sources read on 2026-10-25:

- Sun et al., *LTS: A DASH Streaming System for Dynamic Multi-Layer 3D Gaussian Splatting Scenes*, ACM MMSys 2025, Sections 4.1–4.3 and Eq. (2), DOI https://doi.org/10.1145/3712676.3714445 . Author-hosted paper: https://people.cs.nycu.edu.tw/~chuang/pubs/pdf/2025mmsys.pdf .
- Authors' repository https://github.com/AIINS-NTHU/LTS-DASH-Streaming-System-for-3DGS/blob/main/README.md states the implementation cannot currently be released publicly. No author code is reused here.

The original formulation selects binary **layer–segment** variables. Tile visibility and unbuffered flags determine requested sizes; quality is estimated for layer–segments because Gaussians cross tile boundaries. Lower layers are prerequisites, and an allocation window includes temporal priority. Section 4.3 uses optimal full-space search after visibility checking; it supplies no implementation-level search pseudocode.

This experiment transfers complete candidate exploration, a real resource-budget feasibility check, and maximization over feasible candidates. Each candidate here selects one independent StreamSTGS CRF for every object. The common problem supplies weights and replaces quality increments with measured absolute PSNR. It has one decision epoch, no cached data, no layer prerequisites, and no temporal objective. Even an object with zero weight must receive one CRF to satisfy the common formulation. No camera-based visibility filter is timed or reproduced.

Thus this is **LTS-inspired**, not reproduction of the LTS ABR implementation or its original objective. Its object-level search space is the same K^N set as the brute-force oracle. The implementations are independent: LTS-inspired recursively carries partial sums, while brute-force uses a flat odometer and evaluates complete vectors. Runtime differences reflect these enumeration details and cannot establish a conceptual scalability advantage. Both remain O(N K^N) upper bounds, with exponential candidate counts and O(N) traversal memory; recursive prefix accumulation can reduce repeated summation. Both use identical candidate caps. No paper-reported runtime is reused as a benchmark measurement.

The downloaded PDF and extracted text are local reading aids, excluded by this experiment's .gitignore. The DOI and author URL suffice to reproduce the methodology audit.
