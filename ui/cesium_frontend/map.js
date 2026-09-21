/* Cesium renderer and Streamlit component bridge. No build tooling required. */
(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  let viewer, args, scene, sdkPromise, building = false, visible = false;
  let parentOrigin = '*', revision = '', cameraKey = '', token = null, satellite;
  let useSatellite = false, groups = [], generation = 0;
  const layerVisibility = new Map();
  const domains = new WeakMap();
  const post = (type, extra = {}) => window.parent.postMessage({isStreamlitMessage: true, type, ...extra}, parentOrigin);
  const status = text => { $('status').textContent = text; };
  const valid = (lat, lon) => lat !== null && lon !== null && Number.isFinite(Number(lat)) && Number.isFinite(Number(lon)) && Math.abs(lat) <= 90 && Math.abs(lon) <= 180;
  const at = (value, i, fallback) => Array.isArray(value) ? (value[i] ?? fallback) : (value ?? fallback);
  const plain = value => String(value ?? '').replace(/<br\s*\/?>/gi, '\n').replace(/<[^>]*>/g, '');

  function loadSdk() {
    if (window.Cesium) return Promise.resolve();
    if (sdkPromise) return sdkPromise;
    sdkPromise = new Promise((resolve, reject) => {
      const script = document.createElement('script');
      script.src = 'https://cesium.com/downloads/cesiumjs/releases/1.145/Build/Cesium/Cesium.js';
      const timer = setTimeout(() => reject(new Error('timeout')), 25000);
      script.onload = () => { clearTimeout(timer); resolve(); };
      script.onerror = () => { clearTimeout(timer); reject(new Error('load')); };
      document.head.append(script);
    }).catch(error => { sdkPromise = null; throw error; });
    return sdkPromise;
  }

  function color(css, opacity = 1) {
    const result = Cesium.Color.fromCssColorString(typeof css === 'string' ? css : '#60a5fa') || Cesium.Color.CORNFLOWERBLUE;
    return result.withAlpha(result.alpha * opacity);
  }

  function domain(layer) {
    if (domains.has(layer)) return domains.get(layer);
    const marker = layer.marker || {};
    const values = Array.isArray(marker.color) ? marker.color : layer.z || [];
    const finite = values.filter(v => v !== null && Number.isFinite(Number(v))).map(Number);
    const limits = [marker.cmin ?? layer.zmin ?? (finite.length ? Math.min(...finite) : 0),
                    marker.cmax ?? layer.zmax ?? (finite.length ? Math.max(...finite) : 1)];
    domains.set(layer, limits);
    return limits;
  }

  function scale(layer) {
    const stops = layer.marker?.colorscale || layer.colorscale;
    return Array.isArray(stops) ? stops : [[0, '#440154'], [.25, '#3b528b'], [.5, '#21918c'], [.75, '#5ec962'], [1, '#fde725']];
  }

  function valueColor(layer, value, opacity = 1) {
    if (value === null || value === undefined) return color('#94a3b8', opacity);
    if (typeof value === 'string' && !Number.isFinite(Number(value))) return color(value, opacity);
    const [min, max] = domain(layer), stops = scale(layer);
    const t = Math.max(0, Math.min(1, (Number(value) - min) / (max - min || 1)));
    let hi = stops.findIndex(stop => Number(stop[0]) >= t);
    if (hi < 0) hi = stops.length - 1;
    const lo = Math.max(0, hi - 1), ratio = (t - Number(stops[lo][0])) / (Number(stops[hi][0]) - Number(stops[lo][0]) || 1);
    const result = Cesium.Color.lerp(color(stops[lo][1]), color(stops[hi][1]), ratio, new Cesium.Color());
    return result.withAlpha(result.alpha * opacity);
  }

  function tooltip(layer, i, extras = {}) {
    const values = {lat: layer.lat?.[i], lon: layer.lon?.[i], text: at(layer.text, i, ''),
                    customdata: layer.customdata?.[i], ...extras};
    let text = layer.hovertemplate || layer.name || 'Location';
    text = text.replace(/<extra>(.*?)<\/extra>/gs, (_, content) => content ? '\n' + content : '');
    text = text.replace(/%\{([a-z]+)(?:\[(\d+)\])?(?::([^}]+))?\}/g, (_, name, index, format) => {
      let value = index === undefined ? values[name] : values[name]?.[Number(index)];
      if (value === null || value === undefined) return '—';
      if (format && Number.isFinite(Number(value))) {
        const digits = Number(format.match(/\.(\d+)f/)?.[1] || 0);
        value = Number(value).toLocaleString('en-US', {minimumFractionDigits: digits, maximumFractionDigits: digits, useGrouping: format.includes(',')});
      }
      return String(value);
    });
    return plain(text);
  }

  function addEntity(options, group, info) {
    const entity = viewer.entities.add(options);
    entity.rtm = info;
    entity.show = group.show;
    group.entities.push(entity);
    return entity;
  }

  function polygonHierarchy(rings) {
    const ring = coords => Cesium.Cartesian3.fromDegreesArray(coords.flatMap(p => [Number(p[0]), Number(p[1])]));
    return new Cesium.PolygonHierarchy(ring(rings[0]), rings.slice(1).map(hole => new Cesium.PolygonHierarchy(ring(hole))));
  }

  function drawLayer(layer, index) {
    const group = {entities: [], show: layerVisibility.get(layer.name || String(index)) ?? layer.visible !== false};
    groups.push(group);
    if (layer.geojson) {
      const keyPath = (layer.featureidkey || 'id').split('.');
      const lookup = new Map((layer.locations || []).map((id, i) => [String(id), i]));
      for (const feature of layer.geojson.features || []) {
        const key = keyPath.reduce((obj, key) => obj?.[key], feature);
        const i = lookup.get(String(key));
        if (i === undefined) continue;
        const geometry = feature.geometry;
        const polygons = geometry?.type === 'MultiPolygon' ? geometry.coordinates : geometry?.type === 'Polygon' ? [geometry.coordinates] : [];
        for (const rings of polygons) {
          if (!rings[0]?.length) continue;
          const info = {description: tooltip(layer, i, {location: key, z: layer.z?.[i]})};
          addEntity({polygon: {hierarchy: polygonHierarchy(rings), material: valueColor(layer, layer.z?.[i])}}, group, info);
          const width = layer.marker?.line?.width ?? 1;
          if (width > 0) for (const ring of rings) {
            addEntity({polyline: {positions: Cesium.Cartesian3.fromDegreesArray(ring.flatMap(p => [Number(p[0]), Number(p[1])])),
              width, material: color(layer.marker?.line?.color || '#78a9d7', .8), clampToGround: true}}, group, info);
          }
        }
      }
    } else {
      const mode = layer.mode || 'markers';
      if (mode.includes('lines')) {
        let segment = [];
        const flush = () => {
          if (segment.length > 1) addEntity({polyline: {positions: segment, width: layer.line?.width || 2,
            material: color(layer.line?.color || '#f59e0b'), clampToGround: true}}, group,
            {description: layer.hoverinfo === 'skip' ? '' : plain(layer.name)});
          segment = [];
        };
        (layer.lat || []).forEach((lat, i) => {
          if (valid(lat, layer.lon?.[i])) segment.push(Cesium.Cartesian3.fromDegrees(Number(layer.lon[i]), Number(lat)));
          else flush();
        });
        flush();
      }
      if (mode.includes('markers') || mode.includes('text')) (layer.lat || []).forEach((lat, i) => {
        const lon = layer.lon?.[i];
        if (!valid(lat, lon)) return;
        const marker = layer.marker || {}, options = {position: Cesium.Cartesian3.fromDegrees(Number(lon), Number(lat))};
        // Terrain clamping uses a 3D position cache; projected maps use WGS84 directly.
        const heightReference = viewer.scene.mode === Cesium.SceneMode.SCENE3D
          ? Cesium.HeightReference.CLAMP_TO_GROUND : Cesium.HeightReference.NONE;
        if (mode.includes('markers')) options.point = {pixelSize: Math.max(3, Number(at(marker.size, i, 7))),
          color: valueColor(layer, at(marker.color, i, '#60a5fa'), Number(at(marker.opacity, i, 1))),
          outlineColor: color('#07111f', .65), outlineWidth: 1,
          heightReference, disableDepthTestDistance: Number.POSITIVE_INFINITY};
        if (mode.includes('text')) options.label = {text: plain(at(layer.text, i, '')), font: '12px sans-serif',
          fillColor: Cesium.Color.WHITE, outlineColor: color('#07111f'), outlineWidth: 3,
          style: Cesium.LabelStyle.FILL_AND_OUTLINE, pixelOffset: new Cesium.Cartesian2(0, -18),
          heightReference, disableDepthTestDistance: Number.POSITIVE_INFINITY};
        addEntity(options, group, {layer: index, point: i, description: tooltip(layer, i)});
      });
    }
    if (layer.name) {
      const label = document.createElement('label'); label.className = 'layer';
      const checkbox = document.createElement('input'); checkbox.type = 'checkbox'; checkbox.checked = group.show;
      checkbox.addEventListener('change', () => {
        group.show = checkbox.checked; group.entities.forEach(entity => { entity.show = group.show; });
        layerVisibility.set(layer.name, group.show); viewer.scene.requestRender();
      });
      const dot = document.createElement('span'); dot.className = 'swatch';
      dot.style.background = valueColor(layer, at(layer.marker?.color, 0, layer.z?.[0] ?? '#60a5fa')).toCssColorString();
      label.append(checkbox, dot, document.createTextNode(layer.name)); $('legend').append(label);
    }
    if (layer.showscale || layer.marker?.showscale) {
      const box = document.createElement('div'); box.className = 'scale';
      const bar = document.createElement('div'); bar.className = 'gradient';
      bar.style.background = 'linear-gradient(to right,' + scale(layer).map(s => `${s[1]} ${Number(s[0])*100}%`).join(',') + ')';
      const title = layer.marker?.colorbar?.title || layer.colorbar?.title || 'Score';
      const ends = document.createElement('div'); ends.className = 'endpoints';
      const [min, max] = domain(layer);
      const left = document.createElement('span'), right = document.createElement('span');
      left.textContent = Number(min).toLocaleString('en-US', {maximumFractionDigits: 1});
      right.textContent = Number(max).toLocaleString('en-US', {maximumFractionDigits: 1});
      ends.append(left, right); box.append(document.createTextNode(plain(title.text || title)), bar, ends); $('legend').append(box);
    }
  }

  function drawScene() {
    groups = []; viewer.entities.removeAll(); $('legend').replaceChildren(); $('details').hidden = true;
    viewer.entities.suspendEvents();
    try { scene.layers.forEach(drawLayer); } finally { viewer.entities.resumeEvents(); }
  }

  function home() {
    if (!scene || !viewer) return;
    if (scene.bounds) {
      const [west, south, east, north] = scene.bounds;
      const dx = Math.max((east - west) * .13, .006), dy = Math.max((north - south) * .13, .006);
      viewer.camera.setView({destination: Cesium.Rectangle.fromDegrees(
        Math.max(-180, west-dx), Math.max(-89.9, south-dy), Math.min(180, east+dx), Math.min(89.9, north+dy))});
      viewer.scene.requestRender();
      return;
    }
    const lat = Number(scene.center.lat), lon = Number(scene.center.lon);
    const height = Math.max(350, 40075016.7 * Math.cos(lat * Math.PI/180) / Math.pow(2, Number(scene.zoom)) * 1.6);
    viewer.camera.setView({destination: Cesium.Cartesian3.fromDegrees(lon, lat, height),
      orientation: {heading: 0, pitch: Cesium.Math.toRadians(-90), roll: 0}});
    viewer.scene.requestRender();
  }

  async function configureIon(nextToken) {
    if (token === nextToken) return;
    token = nextToken;
    const current = ++generation;
    satellite = null; useSatellite = false; $('imagery').disabled = true;
    viewer.imageryLayers.removeAll();
    viewer.imageryLayers.addImageryProvider(new Cesium.OpenStreetMapImageryProvider({url: 'https://tile.openstreetmap.org/'}));
    viewer.terrainProvider = new Cesium.EllipsoidTerrainProvider();
    if (!token) { status('Street map · ion token not configured'); return; }
    Cesium.Ion.defaultAccessToken = token;
    status('Connecting to Cesium ion…');
    const outcomes = await Promise.allSettled([
      Cesium.createWorldImageryAsync({style: Cesium.IonWorldImageryStyle.AERIAL_WITH_LABELS}),
      Cesium.createWorldTerrainAsync()
    ]);
    if (current !== generation || !viewer || viewer.isDestroyed()) return;
    if (outcomes[0].status === 'fulfilled') {
      satellite = viewer.imageryLayers.addImageryProvider(outcomes[0].value);
      useSatellite = true; $('imagery').disabled = false; $('imagery').textContent = 'Streets';
      outcomes[0].value.errorEvent.addEventListener(() => {
        satellite.show = false; useSatellite = false; $('imagery').textContent = 'Satellite';
        status('Satellite unavailable · street map active'); viewer.scene.requestRender();
      });
    }
    if (outcomes[1].status === 'fulfilled') viewer.terrainProvider = outcomes[1].value;
    status(outcomes.every(x => x.status === 'fulfilled') ? '' : 'Ion layer unavailable · check token, assets and allowed URLs');
    viewer.scene.requestRender();
  }

  function select(click) {
    const picked = viewer.scene.pick(click.position), info = picked?.id?.rtm;
    if (info?.description) { $('detailText').textContent = info.description; $('details').hidden = false; }
    else $('details').hidden = true;
    if (scene.pick_location) {
      const ray = viewer.camera.getPickRay(click.position);
      const point = (ray && viewer.scene.globe.pick(ray, viewer.scene)) || viewer.camera.pickEllipsoid(click.position);
      if (!point) return;
      const location = Cesium.Cartographic.fromCartesian(point);
      $('detailText').textContent = 'Selected location\n' + Cesium.Math.toDegrees(location.latitude).toFixed(6)
        + ', ' + Cesium.Math.toDegrees(location.longitude).toFixed(6);
      $('details').hidden = false;
      post('streamlit:setComponentValue', {dataType: 'json', value: {revision,
        location: {lat: Cesium.Math.toDegrees(location.latitude), lng: Cesium.Math.toDegrees(location.longitude)}}});
    } else if (args.selectable && Number.isInteger(info?.point)) {
      post('streamlit:setComponentValue', {dataType: 'json', value: {revision, layer: info.layer, point: info.point}});
    }
  }

  async function render() {
    if (!visible || !args || building) return;
    building = true;
    try {
      await loadSdk();
      if (!viewer) {
        viewer = new Cesium.Viewer('map', {animation: false, timeline: false, geocoder: false, homeButton: false,
          sceneModePicker: false, baseLayerPicker: false, navigationHelpButton: false, fullscreenButton: false,
          infoBox: false, selectionIndicator: false, baseLayer: false, requestRenderMode: true,
          maximumRenderTimeChange: Infinity, sceneMode: Cesium.SceneMode.SCENE3D});
        viewer.scene.globe.depthTestAgainstTerrain = false;
        viewer.screenSpaceEventHandler.setInputAction(select, Cesium.ScreenSpaceEventType.LEFT_CLICK);
        viewer.scene.renderError.addEventListener(() => { status('Map rendering failed. Reload or enable browser hardware acceleration.'); });
      }
      viewer.useDefaultRenderLoop = visible;
      scene = args.scene;
      if (revision !== scene.revision) {
        revision = scene.revision;
        drawScene();
        const nextCamera = JSON.stringify([scene.center, scene.zoom, scene.bounds]);
        if (nextCamera !== cameraKey) { cameraKey = nextCamera; home(); }
      }
      viewer.resize(); viewer.scene.requestRender();
      void configureIon(args.token || '');
      $('retry').hidden = true;
    } catch (_) {
      revision = '';
      status('Cesium could not load. Check internet access and WebGL, then retry.'); $('retry').hidden = false;
    } finally { building = false; }
  }

  $('home').onclick = home;
  $('close').onclick = () => { $('details').hidden = true; };
  $('mode').onclick = () => {
    if (!viewer || !scene) return;
    const to2D = viewer.scene.mode !== Cesium.SceneMode.SCENE2D;
    // Finish the projection change before resetting the camera. A morphComplete
    // callback runs before Cesium finishes its own camera reset.
    viewer.entities.removeAll();
    if (to2D) viewer.scene.morphTo2D(0); else viewer.scene.morphTo3D(0);
    drawScene();
    viewer.resize(); home();
    $('mode').textContent = to2D ? '3D' : '2D';
  };
  $('imagery').onclick = () => {
    if (!satellite) return;
    useSatellite = !useSatellite; satellite.show = useSatellite;
    $('imagery').textContent = useSatellite ? 'Streets' : 'Satellite'; viewer.scene.requestRender();
  };
  $('retry').onclick = () => { status('Loading map…'); void render(); };
  window.addEventListener('message', event => {
    if (event.source !== window.parent || event.data?.type !== 'streamlit:render') return;
    parentOrigin = event.origin; args = event.data.args;
    if (!args?.scene) return;
    post('streamlit:setFrameHeight', {height: args.scene.height});
    void render();
  });
  new IntersectionObserver(entries => {
    visible = entries.some(entry => entry.isIntersecting);
    if (viewer) viewer.useDefaultRenderLoop = visible;
    if (visible) void render();
  }).observe(document.body);
  window.addEventListener('pagehide', () => { if (viewer && !viewer.isDestroyed()) viewer.destroy(); });
  post('streamlit:componentReady', {apiVersion: 1});
})();
