import{r as l,_ as ue,F as he,R as pe}from"./monaco-BePFVKFP.js";import{r as me}from"./vendor-CQMrbrXp.js";(function(){const s=document.createElement("link").relList;if(s&&s.supports&&s.supports("modulepreload"))return;for(const n of document.querySelectorAll('link[rel="modulepreload"]'))o(n);new MutationObserver(n=>{for(const r of n)if(r.type==="childList")for(const i of r.addedNodes)i.tagName==="LINK"&&i.rel==="modulepreload"&&o(i)}).observe(document,{childList:!0,subtree:!0});function a(n){const r={};return n.integrity&&(r.integrity=n.integrity),n.referrerPolicy&&(r.referrerPolicy=n.referrerPolicy),n.crossOrigin==="use-credentials"?r.credentials="include":n.crossOrigin==="anonymous"?r.credentials="omit":r.credentials="same-origin",r}function o(n){if(n.ep)return;n.ep=!0;const r=a(n);fetch(n.href,r)}})();var te={exports:{}},F={};/**
 * @license React
 * react-jsx-runtime.production.min.js
 *
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */var ye=l,fe=Symbol.for("react.element"),ge=Symbol.for("react.fragment"),xe=Object.prototype.hasOwnProperty,je=ye.__SECRET_INTERNALS_DO_NOT_USE_OR_YOU_WILL_BE_FIRED.ReactCurrentOwner,ve={key:!0,ref:!0,__self:!0,__source:!0};function se(t,s,a){var o,n={},r=null,i=null;a!==void 0&&(r=""+a),s.key!==void 0&&(r=""+s.key),s.ref!==void 0&&(i=s.ref);for(o in s)xe.call(s,o)&&!ve.hasOwnProperty(o)&&(n[o]=s[o]);if(t&&t.defaultProps)for(o in s=t.defaultProps,s)n[o]===void 0&&(n[o]=s[o]);return{$$typeof:fe,type:t,key:r,ref:i,props:n,_owner:je.current}}F.Fragment=ge;F.jsx=se;F.jsxs=se;te.exports=F;var e=te.exports,q={},J=me;q.createRoot=J.createRoot,q.hydrateRoot=J.hydrateRoot;let M=null,E=null,h=null;const we={},H=async()=>{if(!M)return E||(E=(async()=>{const t=document.createElement("script");t.src="https://cdn.jsdelivr.net/pyodide/v0.24.1/full/pyodide.js",document.head.appendChild(t),await new Promise((s,a)=>{t.onload=()=>s(),t.onerror=()=>a(new Error("Failed to load Pyodide"))}),M=await window.loadPyodide({indexURL:"https://cdn.jsdelivr.net/pyodide/v0.24.1/full/"}),await M.runPythonAsync(`
import sys
import json

class OutputCapture:
    def write(self, text):
        import js
        if hasattr(js, '_pythonOutput'):
            js._pythonOutput(text)
    def flush(self):
        pass

sys.stdout = OutputCapture()
sys.stderr = OutputCapture()
        `),window._pythonOutput=s=>{h==null||h(s)},h==null||h(`✓ Ready
`)})(),E)};let k=null,z=null;const ne=async()=>{if(!k)return z||(z=(async()=>{const{WebR:t}=await ue(async()=>{const{WebR:s}=await import("https://webr.r-wasm.org/latest/webr.mjs");return{WebR:s}},[],import.meta.url);k=new t,await k.init()})(),z)},ae=t=>{const s=t.trim();return s?s.startsWith("#")?"comment":/<-/.test(t)||/^\s*library\s*\(/.test(t)||/^\s*c\s*\(/.test(t)||/\$\w+/.test(t)||/^\s*(cat|print)\s*\(.*\\n/.test(t)?"r":(/^\s*(import|from)\s+/.test(t)||/^\s*def\s+\w+\s*\(/.test(t)||/^\s*class\s+\w+/.test(t)||/print\s*\(f?["']/.test(t)||/\[.*for.*in.*\]/.test(t)||/:\s*$/.test(t),"python"):"empty"},be=t=>{const s=t.split(`
`),a=[];let o=null,n=[],r=1;for(let i=0;i<s.length;i++){const b=s[i],j=b.trim().toLowerCase();if(j==="#%% python"||j==="# python"){n.length>0&&o&&a.push({lang:o,code:n.join(`
`),startLine:r}),o="python",n=[],r=i+2;continue}if(j==="#%% r"||j==="# r"){n.length>0&&o&&a.push({lang:o,code:n.join(`
`),startLine:r}),o="r",n=[],r=i+2;continue}const v=ae(b);v!=="comment"&&v!=="empty"&&(o===null?(o=v,r=i+1):v!==o&&(n.length>0&&a.push({lang:o,code:n.join(`
`),startLine:r}),o=v,n=[],r=i+1)),n.push(b)}if(n.length>0&&o){const i=n.join(`
`).trim();i&&a.push({lang:o,code:i,startLine:r})}return a},Ne=async()=>{if(!(!M||!k))try{const t=await M.runPythonAsync(`
import json
_vars = {}
for name, val in list(globals().items()):
    if not name.startswith('_') and name not in ['sys', 'json', 'OutputCapture']:
        try:
            if isinstance(val, (int, float)):
                _vars[name] = val
            elif isinstance(val, (list, tuple)) and all(isinstance(x, (int, float)) for x in val):
                _vars[name] = list(val)
            elif isinstance(val, str):
                _vars[name] = val
        except:
            pass
json.dumps(_vars)
        `);if(t&&t!=="{}"){const s=JSON.parse(t);for(const[a,o]of Object.entries(s))we[a]=o,typeof o=="number"?await k.evalR(`${a} <- ${o}`):Array.isArray(o)?await k.evalR(`${a} <- c(${o.join(", ")})`):typeof o=="string"&&await k.evalR(`${a} <- "${o}"`)}}catch{}},ke=async t=>{var s;await H();try{await M.runPythonAsync(t)}catch(a){h==null||h(`Error: ${((s=a.message)==null?void 0:s.split(`
`).pop())||a}
`)}},oe=async t=>{var s;await ne(),await Ne();try{const a=`capture.output({ ${t} }, type = "output")`,o=await k.evalR(a);try{const n=await o.toArray();if(n&&n.length>0){const r=n.filter(i=>i!=="").join(`
`);r&&(h==null||h(r+`
`))}}catch{const n=await o.toString();n&&(h==null||h(n+`
`))}}catch(a){h==null||h(`R: ${((s=a.message)==null?void 0:s.split(`
`).slice(-2).join(" "))||a}
`)}},Z=async t=>{const s=be(t);for(const a of s)a.lang==="python"?await ke(a.code):await oe(a.code)},Ce=t=>{const s=t.split(`
`),a=[];let o="py";for(let n=0;n<s.length;n++){const r=s[n].trim().toLowerCase();if(r==="#%% python"||r==="# python")o="py";else if(r==="#%% r"||r==="# r")o="R";else{const i=ae(s[n]);i==="r"?o="R":i==="python"&&(o="py")}s[n].trim()&&!s[n].trim().startsWith("#")&&a.push({line:n+1,lang:o})}return a},re="ai-ide-files";let O=[];const Re=[{name:"script.py",isDirectory:!1,path:"/project/script.py",content:`# Mixed Python + R Code
# Variables are automatically shared!

#%% python  
numbers = [1, 2, 3, 4, 5]
squared = [x ** 2 for x in numbers]
print(f"Numbers: {numbers}")
print(f"Squared: {squared}")

#%% r
cat("Mean:", mean(squared), "\\n")
cat("SD:", sd(squared), "\\n")

#%% python
total = sum(squared)
print(f"Total: {total}")
`}],_e=()=>{try{const t=localStorage.getItem(re);if(t)return JSON.parse(t)}catch{}return[...Re]},Se=()=>{try{localStorage.setItem(re,JSON.stringify(O))}catch{}};O=_e();const W=t=>O.find(s=>s.path===t||s.path==="/project/"+t||s.name===t),Le={selectDirectory:async()=>"/project",readDir:async t=>O.filter(s=>s.path.startsWith(t)).map(s=>({name:s.name,isDirectory:s.isDirectory,path:s.path})),readFile:async t=>{var s;return((s=W(t))==null?void 0:s.content)||""},saveFile:async(t,s)=>{const a=W(t);return a?a.content=s:O.push({name:t.split("/").pop()||t,isDirectory:!1,path:"/project/"+t,content:s}),Se(),{success:!0}}},Me={init:()=>{H().catch(console.error)},send:async t=>{const s=t.trim();if(s.startsWith("python ")||s.includes(".py")){const a=s.match(/python\s+["']?([^"'\s]+)["']?/);if(a){const o=W(a[1]);o!=null&&o.content?await Z(o.content):h==null||h(`File not found
`)}}else s.length>0&&await Z(s)},onData:t=>(h=t,()=>{h=null})},Pe={search:async t=>{try{const a=await(await fetch("./datasets.json")).json();return{success:!0,data:[...a.kaggle.map(n=>({...n,source:"kaggle"})),...a.tensorflow.map(n=>({...n,source:"tensorflow"})),...a.pytorch.map(n=>({...n,source:"pytorch"})),...a.huggingface.map(n=>({...n,source:"huggingface"}))].filter(n=>n.name.toLowerCase().includes(t.toLowerCase())||n.id.toLowerCase().includes(t.toLowerCase())).slice(0,20)}}catch{return{success:!1,data:[]}}}};window.fileSystem=Le;window.terminal=Me;window.appControl={setMode:()=>{}};window.kaggle=Pe;window.analysis={checkDeps:async()=>({missing:[]}),recommend:async()=>({success:!0,recommendation:""})};window.pyodideReady=H();window.runR=oe;window.loadWebR=ne;window.getLanguageAnnotations=Ce;/**
 * @license lucide-react v0.562.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const De=t=>t.replace(/([a-z0-9])([A-Z])/g,"$1-$2").toLowerCase(),$e=t=>t.replace(/^([A-Z])|[\s-_]+(\w)/g,(s,a,o)=>o?o.toUpperCase():a.toLowerCase()),G=t=>{const s=$e(t);return s.charAt(0).toUpperCase()+s.slice(1)},ie=(...t)=>t.filter((s,a,o)=>!!s&&s.trim()!==""&&o.indexOf(s)===a).join(" ").trim(),Ae=t=>{for(const s in t)if(s.startsWith("aria-")||s==="role"||s==="title")return!0};/**
 * @license lucide-react v0.562.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */var Oe={xmlns:"http://www.w3.org/2000/svg",width:24,height:24,viewBox:"0 0 24 24",fill:"none",stroke:"currentColor",strokeWidth:2,strokeLinecap:"round",strokeLinejoin:"round"};/**
 * @license lucide-react v0.562.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const Ee=l.forwardRef(({color:t="currentColor",size:s=24,strokeWidth:a=2,absoluteStrokeWidth:o,className:n="",children:r,iconNode:i,...b},j)=>l.createElement("svg",{ref:j,...Oe,width:s,height:s,stroke:t,strokeWidth:o?Number(a)*24/Number(s):a,className:ie("lucide",n),...!r&&!Ae(b)&&{"aria-hidden":"true"},...b},[...i.map(([v,N])=>l.createElement(v,N)),...Array.isArray(r)?r:[r]]));/**
 * @license lucide-react v0.562.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const w=(t,s)=>{const a=l.forwardRef(({className:o,...n},r)=>l.createElement(Ee,{ref:r,iconNode:s,className:ie(`lucide-${De(G(t))}`,`lucide-${t}`,o),...n}));return a.displayName=G(t),a};/**
 * @license lucide-react v0.562.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const ze=[["path",{d:"M2.97 12.92A2 2 0 0 0 2 14.63v3.24a2 2 0 0 0 .97 1.71l3 1.8a2 2 0 0 0 2.06 0L12 19v-5.5l-5-3-4.03 2.42Z",key:"lc1i9w"}],["path",{d:"m7 16.5-4.74-2.85",key:"1o9zyk"}],["path",{d:"m7 16.5 5-3",key:"va8pkn"}],["path",{d:"M7 16.5v5.17",key:"jnp8gn"}],["path",{d:"M12 13.5V19l3.97 2.38a2 2 0 0 0 2.06 0l3-1.8a2 2 0 0 0 .97-1.71v-3.24a2 2 0 0 0-.97-1.71L17 10.5l-5 3Z",key:"8zsnat"}],["path",{d:"m17 16.5-5-3",key:"8arw3v"}],["path",{d:"m17 16.5 4.74-2.85",key:"8rfmw"}],["path",{d:"M17 16.5v5.17",key:"k6z78m"}],["path",{d:"M7.97 4.42A2 2 0 0 0 7 6.13v4.37l5 3 5-3V6.13a2 2 0 0 0-.97-1.71l-3-1.8a2 2 0 0 0-2.06 0l-3 1.8Z",key:"1xygjf"}],["path",{d:"M12 8 7.26 5.15",key:"1vbdud"}],["path",{d:"m12 8 4.74-2.85",key:"3rx089"}],["path",{d:"M12 13.5V8",key:"1io7kd"}]],K=w("boxes",ze);/**
 * @license lucide-react v0.562.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const Ie=[["path",{d:"M12 18V5",key:"adv99a"}],["path",{d:"M15 13a4.17 4.17 0 0 1-3-4 4.17 4.17 0 0 1-3 4",key:"1e3is1"}],["path",{d:"M17.598 6.5A3 3 0 1 0 12 5a3 3 0 1 0-5.598 1.5",key:"1gqd8o"}],["path",{d:"M17.997 5.125a4 4 0 0 1 2.526 5.77",key:"iwvgf7"}],["path",{d:"M18 18a4 4 0 0 0 2-7.464",key:"efp6ie"}],["path",{d:"M19.967 17.483A4 4 0 1 1 12 18a4 4 0 1 1-7.967-.517",key:"1gq6am"}],["path",{d:"M6 18a4 4 0 0 1-2-7.464",key:"k1g0md"}],["path",{d:"M6.003 5.125a4 4 0 0 0-2.526 5.77",key:"q97ue3"}]],Fe=w("brain",Ie);/**
 * @license lucide-react v0.562.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const Te=[["path",{d:"M5 21v-6",key:"1hz6c0"}],["path",{d:"M12 21V3",key:"1lcnhd"}],["path",{d:"M19 21V9",key:"unv183"}]],Y=w("chart-no-axes-column",Te);/**
 * @license lucide-react v0.562.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const Ve=[["ellipse",{cx:"12",cy:"5",rx:"9",ry:"3",key:"msslwz"}],["path",{d:"M3 5V19A9 3 0 0 0 21 19V5",key:"1wlel7"}],["path",{d:"M3 12A9 3 0 0 0 21 12",key:"mv7ke4"}]],Q=w("database",Ve);/**
 * @license lucide-react v0.562.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const Be=[["path",{d:"M15 3h6v6",key:"1q9fwt"}],["path",{d:"M10 14 21 3",key:"gplh6r"}],["path",{d:"M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6",key:"a6xqqp"}]],Ue=w("external-link",Be);/**
 * @license lucide-react v0.562.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const qe=[["path",{d:"M21 12a9 9 0 1 1-6.219-8.56",key:"13zald"}]],le=w("loader-circle",qe);/**
 * @license lucide-react v0.562.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const We=[["path",{d:"m10 17 5-5-5-5",key:"1bsop3"}],["path",{d:"M15 12H3",key:"6jk70r"}],["path",{d:"M15 3h4a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2h-4",key:"u53s6r"}]],He=w("log-in",We);/**
 * @license lucide-react v0.562.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const Je=[["path",{d:"M5 5a2 2 0 0 1 3.008-1.728l11.997 6.998a2 2 0 0 1 .003 3.458l-12 7A2 2 0 0 1 5 19z",key:"10ikf1"}]],Ze=w("play",Je);/**
 * @license lucide-react v0.562.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const Ge=[["path",{d:"M5 12h14",key:"1ays0h"}],["path",{d:"M12 5v14",key:"s699le"}]],I=w("plus",Ge);/**
 * @license lucide-react v0.562.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const Ke=[["path",{d:"M3 12a9 9 0 0 1 9-9 9.75 9.75 0 0 1 6.74 2.74L21 8",key:"v9h5vc"}],["path",{d:"M21 3v5h-5",key:"1q7to0"}],["path",{d:"M21 12a9 9 0 0 1-9 9 9.75 9.75 0 0 1-6.74-2.74L3 16",key:"3uifl3"}],["path",{d:"M8 16H3v5",key:"1cv678"}]],Ye=w("refresh-cw",Ke);/**
 * @license lucide-react v0.562.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const Qe=[["path",{d:"M15.2 3a2 2 0 0 1 1.4.6l3.8 3.8a2 2 0 0 1 .6 1.4V19a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2z",key:"1c8476"}],["path",{d:"M17 21v-7a1 1 0 0 0-1-1H8a1 1 0 0 0-1 1v7",key:"1ydtos"}],["path",{d:"M7 3v4a1 1 0 0 0 1 1h7",key:"t51u73"}]],Xe=w("save",Qe);/**
 * @license lucide-react v0.562.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const et=[["path",{d:"m21 21-4.34-4.34",key:"14j7rj"}],["circle",{cx:"11",cy:"11",r:"8",key:"4ej97u"}]],tt=w("search",et);/**
 * @license lucide-react v0.562.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const st=[["path",{d:"M9 3H5a2 2 0 0 0-2 2v4m6-6h10a2 2 0 0 1 2 2v4M9 3v18m0 0h10a2 2 0 0 0 2-2V9M9 21H5a2 2 0 0 1-2-2V9m0 0h18",key:"gugj83"}]],X=w("table-2",st);/**
 * @license lucide-react v0.562.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const nt=[["path",{d:"M10 11v6",key:"nco0om"}],["path",{d:"M14 11v6",key:"outv1u"}],["path",{d:"M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6",key:"miytrc"}],["path",{d:"M3 6h18",key:"d0wm0j"}],["path",{d:"M8 6V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2",key:"e791ji"}]],at=w("trash-2",nt);function ot({cell:t,index:s,onUpdate:a,onRun:o,onDelete:n,onAddBelow:r,isActive:i,onFocus:b}){const[j,v]=l.useState(!1),N=l.useRef(null),_=y=>{N.current=y,y.onDidContentSizeChange(()=>{const P=Math.min(400,Math.max(60,y.getContentHeight()));y.layout({height:P,width:y.getLayoutInfo().width})})};return e.jsxs("div",{className:`notebook-cell ${i?"active":""} ${t.isRunning?"running":""}`,onMouseEnter:()=>v(!0),onMouseLeave:()=>v(!1),onClick:()=>b(t.id),children:[e.jsxs("div",{className:"cell-toolbar",children:[e.jsxs("div",{className:"cell-info",children:[e.jsxs("span",{className:"cell-index",children:["[",s+1,"]"]}),e.jsx("span",{className:`cell-lang ${t.language}`,children:t.language})]}),e.jsxs("div",{className:"cell-actions",children:[e.jsx("button",{className:"cell-btn run",onClick:y=>{y.stopPropagation(),o(t.id)},disabled:t.isRunning,title:"Run cell (Shift+Enter)",children:t.isRunning?e.jsx(le,{size:14,className:"spin"}):e.jsx(Ze,{size:14,fill:"currentColor"})}),e.jsx("button",{className:"cell-btn",onClick:y=>{y.stopPropagation(),r(t.id)},title:"Add cell below",children:e.jsx(I,{size:14})}),e.jsx("button",{className:"cell-btn delete",onClick:y=>{y.stopPropagation(),n(t.id)},title:"Delete cell",children:e.jsx(at,{size:14})})]})]}),e.jsx("div",{className:"cell-editor",children:e.jsx(he,{height:"auto",defaultLanguage:t.language==="r"?"r":"python",value:t.content,onChange:y=>a(t.id,y||""),onMount:_,theme:"vs-dark",options:{minimap:{enabled:!1},scrollBeyondLastLine:!1,lineNumbers:"on",lineNumbersMinChars:3,folding:!1,fontSize:13,fontFamily:"'JetBrains Mono', 'SF Mono', Consolas, monospace",padding:{top:8,bottom:8},scrollbar:{vertical:"hidden",horizontal:"auto"},overviewRulerBorder:!1,renderLineHighlight:"none",automaticLayout:!0}})}),t.output&&e.jsx("div",{className:"cell-output",children:e.jsx("pre",{children:t.output})}),j&&e.jsx("div",{className:"add-cell-hint",children:e.jsxs("button",{onClick:()=>r(t.id),children:[e.jsx(I,{size:12})," Add cell"]})})]})}const A=()=>Math.random().toString(36).substr(2,9),ee=[{id:A(),type:"code",language:"python",content:`# 🧠 Deep Learning IDE
# This is a Jupyter-style notebook with Python + R support

import torch
import torch.nn as nn

print("PyTorch version:", torch.__version__)`,output:"",isRunning:!1,isCollapsed:!1},{id:A(),type:"code",language:"python",content:`# Define a simple neural network
class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
    
    def forward(self, x):
        return self.layers(x)

model = SimpleNN()
print(model)`,output:"",isRunning:!1,isCollapsed:!1},{id:A(),type:"code",language:"r",content:`# R Statistical Analysis
data <- c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
cat("Mean:", mean(data), "\\n")
cat("SD:", sd(data), "\\n")
cat("Summary:\\n")
summary(data)`,output:"",isRunning:!1,isCollapsed:!1}];function rt({onRunCell:t,isReady:s}){var $;const[a,o]=l.useState(ee),[n,r]=l.useState((($=ee[0])==null?void 0:$.id)||null),[i,b]=l.useState(!1),[j,v]=l.useState(""),N=l.useCallback((d,m)=>{o(u=>u.map(p=>p.id===d?{...p,content:m}:p))},[]),_=l.useCallback(async d=>{const m=a.find(u=>u.id===d);if(!(!m||!s)){o(u=>u.map(p=>p.id===d?{...p,isRunning:!0,output:""}:p));try{const u=await t(m.content,m.language);o(p=>p.map(f=>f.id===d?{...f,isRunning:!1,output:u}:f))}catch(u){o(p=>p.map(f=>f.id===d?{...f,isRunning:!1,output:`Error: ${u.message}`}:f))}}},[a,t,s]),y=l.useCallback(d=>{a.length<=1||o(m=>m.filter(u=>u.id!==d))},[a.length]),P=l.useCallback(d=>{const m=a.findIndex(f=>f.id===d),u={id:A(),type:"code",language:"python",content:"",output:"",isRunning:!1,isCollapsed:!1},p=[...a];p.splice(m+1,0,u),o(p),r(u.id)},[a]),C=(d="python")=>{const m={id:A(),type:"code",language:d,content:d==="python"?"# Python code":"# R code",output:"",isRunning:!1,isCollapsed:!1};o(u=>[...u,m]),r(m.id)},T=async()=>{for(const d of a)await _(d.id)},D=()=>{const d={nbformat:4,nbformat_minor:5,metadata:{kernelspec:{name:"python3",display_name:"Python 3"}},cells:a.map(f=>({cell_type:"code",source:f.content.split(`
`),metadata:{language:f.language},outputs:f.output?[{output_type:"stream",text:f.output.split(`
`)}]:[]}))},m=new Blob([JSON.stringify(d,null,2)],{type:"application/json"}),u=URL.createObjectURL(m),p=document.createElement("a");p.href=u,p.download="notebook.ipynb",p.click(),URL.revokeObjectURL(u)},V=()=>{alert(`To open in Colab:
1. Save notebook with the button
2. Upload to Google Drive
3. Open with Colaboratory`)},B=()=>{b(!0),v("User")};return e.jsxs("div",{className:"notebook",children:[e.jsxs("div",{className:"notebook-toolbar",children:[e.jsxs("div",{className:"toolbar-left",children:[e.jsxs("button",{className:"toolbar-btn",onClick:()=>C("python"),children:[e.jsx(I,{size:14})," Python"]}),e.jsxs("button",{className:"toolbar-btn",onClick:()=>C("r"),children:[e.jsx(I,{size:14})," R"]}),e.jsx("span",{className:"toolbar-separator"}),e.jsx("button",{className:"toolbar-btn primary",onClick:T,disabled:!s,children:"▶ Run All"})]}),e.jsxs("div",{className:"toolbar-right",children:[e.jsxs("button",{className:"toolbar-btn",onClick:D,children:[e.jsx(Xe,{size:14})," Save .ipynb"]}),e.jsxs("button",{className:"toolbar-btn",onClick:V,children:[e.jsx(Ue,{size:14})," Colab"]}),e.jsx("span",{className:"toolbar-separator"}),i?e.jsxs("span",{className:"user-badge",children:["👤 ",j]}):e.jsxs("button",{className:"toolbar-btn google",onClick:B,children:[e.jsx(He,{size:14})," Sign In"]})]})]}),e.jsxs("div",{className:"notebook-cells",children:[a.map((d,m)=>e.jsx(ot,{cell:d,index:m,onUpdate:N,onRun:_,onDelete:y,onAddBelow:P,isActive:n===d.id,onFocus:r},d.id)),e.jsxs("div",{className:"add-cell-bottom",children:[e.jsx("button",{onClick:()=>C("python"),children:"+ Python"}),e.jsx("button",{onClick:()=>C("r"),children:"+ R"})]})]})]})}const it=[{id:1,name:"Alice",age:28,salary:55e3,dept:"Engineering"},{id:2,name:"Bob",age:34,salary:62e3,dept:"Marketing"},{id:3,name:"Charlie",age:25,salary:48e3,dept:"Engineering"},{id:4,name:"Diana",age:31,salary:71e3,dept:"Sales"},{id:5,name:"Eve",age:29,salary:58e3,dept:"Engineering"}];function lt(){const[t,s]=l.useState(!1),[a,o]=l.useState(!1),[n,r]=l.useState(!1),[i,b]=l.useState(!1),[j,v]=l.useState(!1),[N,_]=l.useState(!1),[y,P]=l.useState([]),[C,T]=l.useState(""),[D,V]=l.useState("all"),[B,$]=l.useState(!1),[d]=l.useState(it),[m,u]=l.useState(""),p=async()=>{$(!0);try{const g=await(await fetch("./datasets.json")).json();P([...g.kaggle.map(x=>({...x,source:"kaggle"})),...g.tensorflow.map(x=>({...x,source:"tensorflow"})),...g.pytorch.map(x=>({...x,source:"pytorch"})),...g.huggingface.map(x=>({...x,source:"huggingface"}))])}catch(c){console.error(c)}$(!1)};l.useEffect(()=>{p(),window.pyodideReady&&window.pyodideReady.then(()=>s(!0)).catch(console.error),window.loadWebR&&window.loadWebR().then(()=>o(!0)).catch(console.error),window.terminal.init()},[]);const f=l.useCallback(async(c,g)=>new Promise(x=>{let S="";const de=window.terminal.onData(L=>{S+=L.replace(/\r\n/g,`
`).replace(/\r/g,`
`)});(async()=>{if(g==="python")window.terminal.send(`${c}
`),await new Promise(L=>setTimeout(L,500));else if(window.runR)try{S=await window.runR(c)||""}catch(L){S=`R Error: ${L.message}`}de(),x(S.trim())})()}),[]),U=async c=>{if(window.runR){u("");try{const g=await window.runR(c);u(g||"")}catch(g){u(`Error: ${g.message}`)}}},R=c=>{console.log("Add layer:",c)},ce=y.filter(c=>{const g=c.name.toLowerCase().includes(C.toLowerCase()),x=D==="all"||c.source===D;return g&&x});return e.jsxs("div",{className:"ide-container",children:[e.jsxs("header",{className:"ide-header",children:[e.jsxs("div",{className:"header-left",children:[e.jsx(Fe,{size:20,className:"logo-icon"}),e.jsx("span",{className:"app-title",children:"Deep Learning IDE"})]}),e.jsxs("div",{className:"header-center",children:[e.jsxs("button",{className:`toggle-btn ${n?"active":""}`,onClick:()=>r(!n),children:[e.jsx(K,{size:14}),e.jsx("span",{children:"Arch"})]}),e.jsxs("button",{className:`toggle-btn ${i?"active":""}`,onClick:()=>b(!i),children:[e.jsx(Q,{size:14}),e.jsx("span",{children:"Data"})]}),e.jsxs("button",{className:`toggle-btn ${j?"active":""}`,onClick:()=>v(!j),children:[e.jsx(X,{size:14}),e.jsx("span",{children:"DF"})]}),e.jsxs("button",{className:`toggle-btn ${N?"active":""}`,onClick:()=>_(!N),children:[e.jsx(Y,{size:14}),e.jsx("span",{children:"R"})]})]}),e.jsx("div",{className:"header-right",children:e.jsxs("div",{className:"status-indicators",children:[e.jsx("span",{className:`status-dot ${t?"ready":"loading"}`,children:"Py"}),e.jsx("span",{className:`status-dot ${a?"ready":"loading"}`,children:"R"})]})})]}),e.jsxs("div",{className:"ide-main",children:[j&&e.jsxs("div",{className:"side-panel left-panel",children:[e.jsxs("div",{className:"panel-header",children:[e.jsx(X,{size:14}),e.jsx("span",{children:"DataFrame"})]}),e.jsxs("div",{className:"panel-content dataframe-panel",children:[e.jsxs("div",{className:"df-info",children:[d.length," × ",Object.keys(d[0]||{}).length]}),e.jsx("div",{className:"dataframe-container",children:e.jsxs("table",{className:"dataframe",children:[e.jsx("thead",{children:e.jsxs("tr",{children:[e.jsx("th",{children:"#"}),Object.keys(d[0]||{}).map(c=>e.jsx("th",{children:c},c))]})}),e.jsx("tbody",{children:d.map((c,g)=>e.jsxs("tr",{children:[e.jsx("td",{className:"row-idx",children:g}),Object.values(c).map((x,S)=>e.jsx("td",{children:typeof x=="number"?x.toLocaleString():x},S))]},g))})]})})]})]}),N&&e.jsxs("div",{className:"side-panel left-panel r-panel",children:[e.jsxs("div",{className:"panel-header",children:[e.jsx(Y,{size:14}),e.jsx("span",{children:"R Console"}),a&&e.jsx("span",{className:"ready-dot",children:"●"})]}),e.jsxs("div",{className:"panel-content",children:[e.jsxs("div",{className:"r-section",children:[e.jsx("h4",{children:"Quick Stats"}),e.jsx("button",{className:"r-btn",onClick:()=>U("x <- c(1,2,3,4,5); mean(x)"),children:"mean()"}),e.jsx("button",{className:"r-btn",onClick:()=>U("x <- c(1,2,3,4,5); sd(x)"),children:"sd()"}),e.jsx("button",{className:"r-btn",onClick:()=>U("x <- c(1,2,3,4,5); summary(x)"),children:"summary()"})]}),m&&e.jsx("div",{className:"r-output",children:e.jsx("pre",{children:m})})]})]}),e.jsxs("div",{className:"center-panel",children:[n&&e.jsxs("div",{className:"architecture-panel",children:[e.jsxs("div",{className:"panel-header",children:[e.jsx(K,{size:14}),e.jsx("span",{children:"PyTorch Layers"})]}),e.jsx("div",{className:"architecture-content",children:e.jsxs("div",{className:"layer-palette",children:[e.jsx("button",{className:"layer-btn linear",onClick:()=>R("nn.Linear(in, out)"),children:"Linear"}),e.jsx("button",{className:"layer-btn conv",onClick:()=>R("nn.Conv2d(in, out, 3)"),children:"Conv2d"}),e.jsx("button",{className:"layer-btn pool",onClick:()=>R("nn.MaxPool2d(2)"),children:"MaxPool"}),e.jsx("button",{className:"layer-btn norm",onClick:()=>R("nn.BatchNorm2d(ch)"),children:"BatchNorm"}),e.jsx("button",{className:"layer-btn rnn",onClick:()=>R("nn.LSTM(in, hidden)"),children:"LSTM"}),e.jsx("button",{className:"layer-btn act",onClick:()=>R("nn.ReLU()"),children:"ReLU"}),e.jsx("button",{className:"layer-btn act",onClick:()=>R("nn.Dropout(0.5)"),children:"Dropout"})]})})]}),e.jsx(rt,{onRunCell:f,isReady:t})]}),i&&e.jsxs("div",{className:"side-panel right-panel database-panel",children:[e.jsxs("div",{className:"panel-header",children:[e.jsx(Q,{size:14}),e.jsx("span",{children:"Datasets"}),e.jsx("button",{className:"icon-btn",onClick:p,children:e.jsx(Ye,{size:11})})]}),e.jsxs("div",{className:"panel-content",children:[e.jsxs("div",{className:"search-box",children:[e.jsx(tt,{size:12}),e.jsx("input",{placeholder:"Search...",value:C,onChange:c=>T(c.target.value)})]}),e.jsx("div",{className:"filter-tabs",children:["all","kaggle","pytorch","tensorflow","huggingface"].map(c=>e.jsx("button",{className:`filter-tab ${D===c?"active":""}`,onClick:()=>V(c),children:c},c))}),e.jsx("div",{className:"dataset-list",children:B?e.jsx("div",{className:"loading",children:e.jsx(le,{size:14,className:"spin"})}):ce.slice(0,20).map(c=>e.jsxs("button",{className:"dataset-btn",children:[e.jsx("span",{className:"ds-name",children:c.name}),e.jsx("span",{className:"ds-meta",children:c.source})]},c.id+c.source))})]})]})]}),e.jsxs("footer",{className:"ide-footer",children:[e.jsx("span",{children:"🧠 Deep Learning IDE"}),e.jsx("span",{className:"footer-separator",children:"|"}),e.jsxs("span",{children:["Python ",t?"✓":"..."]}),e.jsx("span",{className:"footer-separator",children:"|"}),e.jsxs("span",{children:["R ",a?"✓":"..."]}),e.jsx("span",{className:"footer-right",children:"PyTorch + WebR"})]})]})}q.createRoot(document.getElementById("root")).render(e.jsx(pe.StrictMode,{children:e.jsx(lt,{})}));
