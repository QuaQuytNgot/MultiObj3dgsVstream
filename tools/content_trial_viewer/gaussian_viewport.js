import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { SplatMesh, SparkRenderer } from "./vendor/spark/dist/spark.module.js";

// Lazy-loaded by the existing PNG shell. One mesh is present at any instant.
export class GaussianViewport {
  constructor(container, onStatus) {
    this.container = container;
    this.onStatus = onStatus;
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0, 0, 0);
    this.camera = new THREE.PerspectiveCamera(39.5978, 1, 0.01, 100);
    this.camera.up.set(0, 0, 1);
    this.renderer = new THREE.WebGLRenderer({ antialias: false, preserveDrawingBuffer: true });
    this.renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    this.renderer.setClearColor(0x000000, 1);
    this.renderer.domElement.id = "gaussianCanvas";
    this.renderer.domElement.setAttribute("aria-label", "Decoded Gaussian free-camera viewport");
    container.replaceChildren(this.renderer.domElement);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = false;
    this.controls.minDistance = 0.05;
    this.controls.maxDistance = 50;
    this.renderer.domElement.addEventListener("contextmenu", event => event.preventDefault());
    this.spark = new SparkRenderer({ renderer: this.renderer, enableLod: false });
    this.scene.add(this.spark);
    this.mesh = null;
    this.generation = 0;
    this.activeLoads = 0;
    this.pending = null;
    this.loading = false;
    this.disposedMeshes = 0;
    this.loaded = null;
    this.object = null;
    this.settings = {};
    this.resetCamera();
    this.observer = new ResizeObserver(() => this.resize());
    this.observer.observe(container);
    this.resize();
    this.renderer.setAnimationLoop(() => {
      this.controls.update();
      this.renderer.render(this.scene, this.camera);
    });
    window.contentGaussianViewport = this;
  }

  resize() {
    const width = Math.max(1, this.container.clientWidth);
    const height = Math.max(1, this.container.clientHeight);
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(width, height, false);
  }

  resetCamera() {
    const center = this.settings.center || [0, 0, -0.1];
    this.controls.target.fromArray(center);
    const distance = this.settings.distance || 4.0311287;
    this.camera.position.set(center[0], center[1] - distance, center[2] + 0.15);
    this.camera.fov = this.settings.fov_degrees || 39.5978;
    this.camera.up.set(...(this.settings.up_axis === "y" ? [0, 1, 0] : [0, 0, 1]));
    this.camera.lookAt(this.controls.target);
    this.camera.updateProjectionMatrix();
    this.controls.update();
  }

  clearMesh() {
    if (this.mesh) {
      this.scene.remove(this.mesh);
      this.mesh.dispose();
      this.mesh = null;
      this.disposedMeshes++;
    }
    this.loaded = null;
  }

  select(asset, url, settings) {
    const generation = ++this.generation;
    this.pending = asset ? { asset, url, settings, generation } : null;
    this.clearMesh();
    this.onStatus({ status: asset ? "loading" : "empty", asset });
    // A running WASM load is allowed to finish and is then disposed before the
    // latest pending selection is created. Rapid switches never pile up models.
    if (!this.loading && this.pending) this.loadPending();
  }

  async loadPending() {
    const selected = this.pending;
    if (!selected) return;
    this.pending = null;
    this.loading = true;
    this.activeLoads++;
    let mesh;
    try {
      mesh = new SplatMesh({ url: selected.url, extSplats: true, lod: false, enableLod: false, nonLod: true });
      await mesh.initialized;
      if (selected.generation !== this.generation) {
        mesh.dispose();
        this.disposedMeshes++;
        mesh = null;
        return;
      }
      if (mesh.numSplats !== selected.asset.gaussian_count) throw new Error("Loaded Gaussian count differs from decoded asset metadata");
      this.clearMesh();
      this.mesh = mesh;
      this.scene.add(mesh);
      this.settings = selected.settings || {};
      if (this.object !== selected.asset.object) {
        this.object = selected.asset.object;
        this.resetCamera();
      }
      this.loaded = { ...selected.asset, url: selected.url };
      await this.spark.update({ scene: this.scene, camera: this.camera });
      if (selected.generation === this.generation) this.onStatus({ status: "loaded", asset: selected.asset });
    } catch (error) {
      if (mesh && mesh !== this.mesh) {
        mesh.dispose();
        this.disposedMeshes++;
      }
      if (selected.generation === this.generation) this.onStatus({ status: "error", error: error.message });
    } finally {
      this.activeLoads--;
      this.loading = false;
      if (this.pending) this.loadPending();
    }
  }

  snapshot() {
    return { loaded: this.loaded, activeLoads: this.activeLoads, pending: !!this.pending,
      meshCount: this.mesh ? 1 : 0, disposedMeshes: this.disposedMeshes,
      cameraPosition: this.camera.position.toArray(), target: this.controls.target.toArray(),
      lod: false, extSplats: true, backend: "SparkJS 2.2.0" };
  }

  nonblackPixels() {
    this.renderer.render(this.scene, this.camera);
    const gl = this.renderer.getContext();
    const pixels = new Uint8Array(gl.drawingBufferWidth * gl.drawingBufferHeight * 4);
    gl.readPixels(0, 0, gl.drawingBufferWidth, gl.drawingBufferHeight, gl.RGBA, gl.UNSIGNED_BYTE, pixels);
    let count = 0;
    for (let i = 0; i < pixels.length; i += 4) if (pixels[i] + pixels[i + 1] + pixels[i + 2] > 15) count++;
    return count;
  }

  dispose() {
    this.generation++;
    this.pending = null;
    this.clearMesh();
    this.observer.disconnect();
    this.renderer.setAnimationLoop(null);
    this.controls.dispose();
    this.spark.dispose();
    this.renderer.dispose();
  }
}
