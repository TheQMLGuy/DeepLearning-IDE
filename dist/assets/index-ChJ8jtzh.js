import{r as d,_ as Ne,F as be,R as ke}from"./monaco-BePFVKFP.js";import{r as Ce}from"./vendor-CQMrbrXp.js";(function(){const t=document.createElement("link").relList;if(t&&t.supports&&t.supports("modulepreload"))return;for(const n of document.querySelectorAll('link[rel="modulepreload"]'))a(n);new MutationObserver(n=>{for(const i of n)if(i.type==="childList")for(const c of i.addedNodes)c.tagName==="LINK"&&c.rel==="modulepreload"&&a(c)}).observe(document,{childList:!0,subtree:!0});function r(n){const i={};return n.integrity&&(i.integrity=n.integrity),n.referrerPolicy&&(i.referrerPolicy=n.referrerPolicy),n.crossOrigin==="use-credentials"?i.credentials="include":n.crossOrigin==="anonymous"?i.credentials="omit":i.credentials="same-origin",i}function a(n){if(n.ep)return;n.ep=!0;const i=r(n);fetch(n.href,i)}})();var X={exports:{}},$={};/**
 * @license React
 * react-jsx-runtime.production.min.js
 *
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */var _e=d,Re=Symbol.for("react.element"),Se=Symbol.for("react.fragment"),Le=Object.prototype.hasOwnProperty,$e=_e.__SECRET_INTERNALS_DO_NOT_USE_OR_YOU_WILL_BE_FIRED.ReactCurrentOwner,De={key:!0,ref:!0,__self:!0,__source:!0};function ee(s,t,r){var a,n={},i=null,c=null;r!==void 0&&(i=""+r),t.key!==void 0&&(i=""+t.key),t.ref!==void 0&&(c=t.ref);for(a in t)Le.call(t,a)&&!De.hasOwnProperty(a)&&(n[a]=t[a]);if(s&&s.defaultProps)for(a in t=s.defaultProps,t)n[a]===void 0&&(n[a]=t[a]);return{$$typeof:Re,type:s,key:i,ref:c,props:n,_owner:$e.current}}$.Fragment=Se;$.jsx=ee;$.jsxs=ee;X.exports=$;var e=X.exports,F={},J=Ce;F.createRoot=J.createRoot,F.hydrateRoot=J.hydrateRoot;let b=null,S=null,h=null;const Pe={},I=async()=>{if(!b)return S||(S=(async()=>{const s=document.createElement("script");s.src="https://cdn.jsdelivr.net/pyodide/v0.24.1/full/pyodide.js",document.head.appendChild(s),await new Promise((t,r)=>{s.onload=()=>t(),s.onerror=()=>r(new Error("Failed to load Pyodide"))}),b=await window.loadPyodide({indexURL:"https://cdn.jsdelivr.net/pyodide/v0.24.1/full/"}),await b.runPythonAsync(`
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
        `),window._pythonOutput=t=>{h==null||h(t)},h==null||h(`✓ Ready
`)})(),S)};let g=null,L=null;const te=async()=>{if(!g)return L||(L=(async()=>{const{WebR:s}=await Ne(async()=>{const{WebR:t}=await import("https://webr.r-wasm.org/latest/webr.mjs");return{WebR:t}},[],import.meta.url);g=new s,await g.init()})(),L)},se=s=>{const t=s.trim();return t?t.startsWith("#")?"comment":/<-/.test(s)||/^\s*library\s*\(/.test(s)||/^\s*c\s*\(/.test(s)||/\$\w+/.test(s)||/^\s*(cat|print)\s*\(.*\\n/.test(s)?"r":(/^\s*(import|from)\s+/.test(s)||/^\s*def\s+\w+\s*\(/.test(s)||/^\s*class\s+\w+/.test(s)||/print\s*\(f?["']/.test(s)||/\[.*for.*in.*\]/.test(s)||/:\s*$/.test(s),"python"):"empty"},Ae=s=>{const t=s.split(`
`),r=[];let a=null,n=[],i=1;for(let c=0;c<t.length;c++){const x=t[c],m=x.trim().toLowerCase();if(m==="#%% python"||m==="# python"){n.length>0&&a&&r.push({lang:a,code:n.join(`
`),startLine:i}),a="python",n=[],i=c+2;continue}if(m==="#%% r"||m==="# r"){n.length>0&&a&&r.push({lang:a,code:n.join(`
`),startLine:i}),a="r",n=[],i=c+2;continue}const f=se(x);f!=="comment"&&f!=="empty"&&(a===null?(a=f,i=c+1):f!==a&&(n.length>0&&r.push({lang:a,code:n.join(`
`),startLine:i}),a=f,n=[],i=c+1)),n.push(x)}if(n.length>0&&a){const c=n.join(`
`).trim();c&&r.push({lang:a,code:c,startLine:i})}return r},Me=async()=>{if(!(!b||!g))try{const s=await b.runPythonAsync(`
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
        `);if(s&&s!=="{}"){const t=JSON.parse(s);for(const[r,a]of Object.entries(t))Pe[r]=a,typeof a=="number"?await g.evalR(`${r} <- ${a}`):Array.isArray(a)?await g.evalR(`${r} <- c(${a.join(", ")})`):typeof a=="string"&&await g.evalR(`${r} <- "${a}"`)}}catch{}},Oe=async s=>{var t;await I();try{await b.runPythonAsync(s)}catch(r){h==null||h(`Error: ${((t=r.message)==null?void 0:t.split(`
`).pop())||r}
`)}},ae=async s=>{var t;await te(),await Me();try{const r=`capture.output({ ${s} }, type = "output")`,a=await g.evalR(r);try{const n=await a.toArray();if(n&&n.length>0){const i=n.filter(c=>c!=="").join(`
`);i&&(h==null||h(i+`
`))}}catch{const n=await a.toString();n&&(h==null||h(n+`
`))}}catch(r){h==null||h(`R: ${((t=r.message)==null?void 0:t.split(`
`).slice(-2).join(" "))||r}
`)}},K=async s=>{const t=Ae(s);for(const r of t)r.lang==="python"?await Oe(r.code):await ae(r.code)},Ee=s=>{const t=s.split(`
`),r=[];let a="py";for(let n=0;n<t.length;n++){const i=t[n].trim().toLowerCase();if(i==="#%% python"||i==="# python")a="py";else if(i==="#%% r"||i==="# r")a="R";else{const c=se(t[n]);c==="r"?a="R":c==="python"&&(a="py")}t[n].trim()&&!t[n].trim().startsWith("#")&&r.push({line:n+1,lang:a})}return r},ne="ai-ide-files";let R=[];const ze=[{name:"script.py",isDirectory:!1,path:"/project/script.py",content:`# Mixed Python + R Code
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
`}],Fe=()=>{try{const s=localStorage.getItem(ne);if(s)return JSON.parse(s)}catch{}return[...ze]},Te=()=>{try{localStorage.setItem(ne,JSON.stringify(R))}catch{}};R=Fe();const T=s=>R.find(t=>t.path===s||t.path==="/project/"+s||t.name===s),Ie={selectDirectory:async()=>"/project",readDir:async s=>R.filter(t=>t.path.startsWith(s)).map(t=>({name:t.name,isDirectory:t.isDirectory,path:t.path})),readFile:async s=>{var t;return((t=T(s))==null?void 0:t.content)||""},saveFile:async(s,t)=>{const r=T(s);return r?r.content=t:R.push({name:s.split("/").pop()||s,isDirectory:!1,path:"/project/"+s,content:t}),Te(),{success:!0}}},Ve={init:()=>{I().catch(console.error)},send:async s=>{const t=s.trim();if(t.startsWith("python ")||t.includes(".py")){const r=t.match(/python\s+["']?([^"'\s]+)["']?/);if(r){const a=T(r[1]);a!=null&&a.content?await K(a.content):h==null||h(`File not found
`)}}else t.length>0&&await K(t)},onData:s=>(h=s,()=>{h=null})},qe={search:async s=>{try{const r=await(await fetch("./datasets.json")).json();return{success:!0,data:[...r.kaggle.map(n=>({...n,source:"kaggle"})),...r.tensorflow.map(n=>({...n,source:"tensorflow"})),...r.pytorch.map(n=>({...n,source:"pytorch"})),...r.huggingface.map(n=>({...n,source:"huggingface"}))].filter(n=>n.name.toLowerCase().includes(s.toLowerCase())||n.id.toLowerCase().includes(s.toLowerCase())).slice(0,20)}}catch{return{success:!1,data:[]}}}};window.fileSystem=Ie;window.terminal=Ve;window.appControl={setMode:()=>{}};window.kaggle=qe;window.analysis={checkDeps:async()=>({missing:[]}),recommend:async()=>({success:!0,recommendation:""})};window.pyodideReady=I();window.runR=ae;window.loadWebR=te;window.getLanguageAnnotations=Ee;/**
 * @license lucide-react v0.562.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const Be=s=>s.replace(/([a-z0-9])([A-Z])/g,"$1-$2").toLowerCase(),We=s=>s.replace(/^([A-Z])|[\s-_]+(\w)/g,(t,r,a)=>a?a.toUpperCase():r.toLowerCase()),Z=s=>{const t=We(s);return t.charAt(0).toUpperCase()+t.slice(1)},re=(...s)=>s.filter((t,r,a)=>!!t&&t.trim()!==""&&a.indexOf(t)===r).join(" ").trim(),Ue=s=>{for(const t in s)if(t.startsWith("aria-")||t==="role"||t==="title")return!0};/**
 * @license lucide-react v0.562.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */var Je={xmlns:"http://www.w3.org/2000/svg",width:24,height:24,viewBox:"0 0 24 24",fill:"none",stroke:"currentColor",strokeWidth:2,strokeLinecap:"round",strokeLinejoin:"round"};/**
 * @license lucide-react v0.562.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const Ke=d.forwardRef(({color:s="currentColor",size:t=24,strokeWidth:r=2,absoluteStrokeWidth:a,className:n="",children:i,iconNode:c,...x},m)=>d.createElement("svg",{ref:m,...Je,width:t,height:t,stroke:s,strokeWidth:a?Number(r)*24/Number(t):r,className:re("lucide",n),...!i&&!Ue(x)&&{"aria-hidden":"true"},...x},[...c.map(([f,D])=>d.createElement(f,D)),...Array.isArray(i)?i:[i]]));/**
 * @license lucide-react v0.562.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const y=(s,t)=>{const r=d.forwardRef(({className:a,...n},i)=>d.createElement(Ke,{ref:i,iconNode:t,className:re(`lucide-${Be(Z(s))}`,`lucide-${s}`,a),...n}));return r.displayName=Z(s),r};/**
 * @license lucide-react v0.562.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const Ze=[["path",{d:"M2.97 12.92A2 2 0 0 0 2 14.63v3.24a2 2 0 0 0 .97 1.71l3 1.8a2 2 0 0 0 2.06 0L12 19v-5.5l-5-3-4.03 2.42Z",key:"lc1i9w"}],["path",{d:"m7 16.5-4.74-2.85",key:"1o9zyk"}],["path",{d:"m7 16.5 5-3",key:"va8pkn"}],["path",{d:"M7 16.5v5.17",key:"jnp8gn"}],["path",{d:"M12 13.5V19l3.97 2.38a2 2 0 0 0 2.06 0l3-1.8a2 2 0 0 0 .97-1.71v-3.24a2 2 0 0 0-.97-1.71L17 10.5l-5 3Z",key:"8zsnat"}],["path",{d:"m17 16.5-5-3",key:"8arw3v"}],["path",{d:"m17 16.5 4.74-2.85",key:"8rfmw"}],["path",{d:"M17 16.5v5.17",key:"k6z78m"}],["path",{d:"M7.97 4.42A2 2 0 0 0 7 6.13v4.37l5 3 5-3V6.13a2 2 0 0 0-.97-1.71l-3-1.8a2 2 0 0 0-2.06 0l-3 1.8Z",key:"1xygjf"}],["path",{d:"M12 8 7.26 5.15",key:"1vbdud"}],["path",{d:"m12 8 4.74-2.85",key:"3rx089"}],["path",{d:"M12 13.5V8",key:"1io7kd"}]],G=y("boxes",Ze);/**
 * @license lucide-react v0.562.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const Ge=[["path",{d:"M5 21v-6",key:"1hz6c0"}],["path",{d:"M12 21V3",key:"1lcnhd"}],["path",{d:"M19 21V9",key:"unv183"}]],H=y("chart-no-axes-column",Ge);/**
 * @license lucide-react v0.562.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const He=[["path",{d:"M20 6 9 17l-5-5",key:"1gmf2c"}]],Ye=y("check",He);/**
 * @license lucide-react v0.562.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const Qe=[["rect",{width:"14",height:"14",x:"8",y:"8",rx:"2",ry:"2",key:"17jyea"}],["path",{d:"M4 16c-1.1 0-2-.9-2-2V4c0-1.1.9-2 2-2h10c1.1 0 2 .9 2 2",key:"zix9uf"}]],Xe=y("copy",Qe);/**
 * @license lucide-react v0.562.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const et=[["ellipse",{cx:"12",cy:"5",rx:"9",ry:"3",key:"msslwz"}],["path",{d:"M3 5V19A9 3 0 0 0 21 19V5",key:"1wlel7"}],["path",{d:"M3 12A9 3 0 0 0 21 12",key:"mv7ke4"}]],Y=y("database",et);/**
 * @license lucide-react v0.562.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const tt=[["path",{d:"M21 12a9 9 0 1 1-6.219-8.56",key:"13zald"}]],_=y("loader-circle",tt);/**
 * @license lucide-react v0.562.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const st=[["path",{d:"M5 5a2 2 0 0 1 3.008-1.728l11.997 6.998a2 2 0 0 1 .003 3.458l-12 7A2 2 0 0 1 5 19z",key:"10ikf1"}]],at=y("play",st);/**
 * @license lucide-react v0.562.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const nt=[["path",{d:"M3 12a9 9 0 0 1 9-9 9.75 9.75 0 0 1 6.74 2.74L21 8",key:"v9h5vc"}],["path",{d:"M21 3v5h-5",key:"1q7to0"}],["path",{d:"M21 12a9 9 0 0 1-9 9 9.75 9.75 0 0 1-6.74-2.74L3 16",key:"3uifl3"}],["path",{d:"M8 16H3v5",key:"1cv678"}]],rt=y("refresh-cw",nt);/**
 * @license lucide-react v0.562.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const ot=[["path",{d:"M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8",key:"1357e3"}],["path",{d:"M3 3v5h5",key:"1xhq8a"}]],it=y("rotate-ccw",ot);/**
 * @license lucide-react v0.562.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const ct=[["path",{d:"m21 21-4.34-4.34",key:"14j7rj"}],["circle",{cx:"11",cy:"11",r:"8",key:"4ej97u"}]],lt=y("search",ct);/**
 * @license lucide-react v0.562.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const dt=[["path",{d:"M9 3H5a2 2 0 0 0-2 2v4m6-6h10a2 2 0 0 1 2 2v4M9 3v18m0 0h10a2 2 0 0 0 2-2V9M9 21H5a2 2 0 0 1-2-2V9m0 0h18",key:"gugj83"}]],Q=y("table-2",dt),ht=({initialValue:s="# Write your code here",language:t="python",onChange:r,theme:a="vs-dark"})=>e.jsx("div",{style:{width:"100%",height:"100%",overflow:"hidden"},children:e.jsx(be,{height:"100%",defaultLanguage:t,defaultValue:s,theme:a,onChange:r,options:{minimap:{enabled:!1},fontSize:14,scrollBeyondLastLine:!1,automaticLayout:!0,padding:{top:16},fontFamily:"'JetBrains Mono', 'Fira Code', Consolas, monospace"}})}),pt=[{id:1,name:"Alice",age:28,salary:55e3,dept:"Engineering"},{id:2,name:"Bob",age:34,salary:62e3,dept:"Marketing"},{id:3,name:"Charlie",age:25,salary:48e3,dept:"Engineering"},{id:4,name:"Diana",age:31,salary:71e3,dept:"Sales"},{id:5,name:"Eve",age:29,salary:58e3,dept:"Engineering"}];function ut(){const[s,t]=d.useState(`# Python + R Notebook
# Variables are automatically shared between languages!

#%% python
numbers = [1, 2, 3, 4, 5]
squared = [x ** 2 for x in numbers]
print(f"Numbers: {numbers}")
print(f"Squared: {squared}")

#%% r
cat("Mean:", mean(squared), "\\n")
cat("SD:", sd(squared), "\\n")

#%% python
print(f"Sum: {sum(squared)}")
`),[r,a]=d.useState(""),[n,i]=d.useState(!1),[c,x]=d.useState(!1),[m,f]=d.useState(!1),[D,V]=d.useState(!1),[P,oe]=d.useState(!1),[A,ie]=d.useState(!1),[M,ce]=d.useState(!1),[w,le]=d.useState(!1),[de,he]=d.useState([]),[O,pe]=d.useState(""),[E,ue]=d.useState("all"),[me,q]=d.useState(!1),[k,ye]=d.useState(pt),[v,fe]=d.useState(null),B=async()=>{q(!0);try{const l=await(await fetch("./datasets.json")).json(),p=[...l.kaggle.map(u=>({...u,source:"kaggle"})),...l.tensorflow.map(u=>({...u,source:"tensorflow"})),...l.pytorch.map(u=>({...u,source:"pytorch"})),...l.huggingface.map(u=>({...u,source:"huggingface"}))];he(p)}catch(o){console.error("Failed to load datasets",o)}q(!1)};d.useEffect(()=>{B(),window.pyodideReady&&window.pyodideReady.then(()=>{x(!0),a(`✓ Python ready
`)}).catch(l=>a(`Error: ${l}
`));const o=window.terminal.onData(l=>{const p=l.replace(/\r\n/g,`
`).replace(/\r/g,`
`);(p.trim()||p.includes(`
`))&&a(u=>u+p)});return window.terminal.init(),o},[]),d.useEffect(()=>{w&&!m&&window.loadWebR&&window.loadWebR().then(()=>f(!0)).catch(console.error)},[w,m]);const xe=async()=>{!c||n||(i(!0),a(""),await window.fileSystem.saveFile("script.py",s),window.terminal.send(`python "script.py"
`),setTimeout(()=>i(!1),800))},N=async o=>{if(window.runR){a(l=>l+`
[R] Running...
`);try{const l=await window.runR(o);a(p=>p+l+`
`)}catch(l){a(p=>p+`R Error: ${l.message}
`)}}},je=()=>a(""),ge=()=>{navigator.clipboard.writeText(r),V(!0),setTimeout(()=>V(!1),2e3)},we=(o,l,p)=>{const u=[...k],C=parseFloat(p);u[o][l]=isNaN(C)?p:C,ye(u);const U=`
# DataFrame updated
import pandas as pd
data = ${JSON.stringify(u,null,2)}
df = pd.DataFrame(data)
print(df)`;t(z=>z.includes("# DataFrame updated")?z.replace(/# DataFrame updated[\s\S]*?print\(df\)/,U.trim()):z+U)},j=o=>{t(l=>l+`
${o}`)},W=de.filter(o=>{const l=o.name.toLowerCase().includes(O.toLowerCase())||o.id.toLowerCase().includes(O.toLowerCase()),p=E==="all"||o.source===E;return l&&p}),ve=o=>{switch(o.source){case"pytorch":return`# ${o.name} Dataset
import torch
from torchvision import datasets, transforms

transform = transforms.Compose([transforms.ToTensor()])
train_data = datasets.${o.id}(root='./data', train=True, download=True, transform=transform)
test_data = datasets.${o.id}(root='./data', train=False, download=True, transform=transform)

print(f"Training samples: {len(train_data)}")
print(f"Test samples: {len(test_data)}")
`;case"tensorflow":return`# ${o.name} Dataset
import tensorflow as tf

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.${o.id}.load_data()

print(f"Training: {x_train.shape}")
print(f"Test: {x_test.shape}")
`;case"huggingface":return`# ${o.name} Dataset
from datasets import load_dataset

dataset = load_dataset("${o.id}")
print(dataset)
`;default:return`# ${o.name} from Kaggle
# Dataset ID: ${o.id}
import pandas as pd

# Download from Kaggle:
# kaggle datasets download -d ${o.id}

print("Dataset: ${o.name}")
print("Category: ${o.category||"N/A"}")
print("Size: ${o.size||"N/A"}")
`}};return e.jsxs("div",{className:"ide-container",children:[e.jsxs("header",{className:"ide-header",children:[e.jsxs("div",{className:"header-left",children:[e.jsx("span",{className:"app-title",children:"🔥 AI IDE"}),e.jsx("span",{className:"badge pytorch",children:"PyTorch"}),e.jsx("span",{className:"badge r-badge",children:"R"})]}),e.jsxs("div",{className:"header-center",children:[e.jsxs("button",{className:`toggle-btn ${P?"active":""}`,onClick:()=>oe(!P),children:[e.jsx(G,{size:15}),e.jsx("span",{children:"Architecture"})]}),e.jsxs("button",{className:`toggle-btn ${A?"active":""}`,onClick:()=>ie(!A),children:[e.jsx(Y,{size:15}),e.jsx("span",{children:"Database"})]}),e.jsxs("button",{className:`toggle-btn ${M?"active":""}`,onClick:()=>ce(!M),children:[e.jsx(Q,{size:15}),e.jsx("span",{children:"Pandas"})]}),e.jsxs("button",{className:`toggle-btn ${w?"active":""}`,onClick:()=>le(!w),children:[e.jsx(H,{size:15}),e.jsx("span",{children:"R"})]})]}),e.jsxs("div",{className:"header-right",children:[!c&&e.jsxs("div",{className:"loading-indicator",children:[e.jsx(_,{size:14,className:"spin"}),e.jsx("span",{children:"Python..."})]}),w&&!m&&e.jsxs("div",{className:"loading-indicator",children:[e.jsx(_,{size:14,className:"spin"}),e.jsx("span",{children:"R..."})]}),e.jsxs("button",{onClick:xe,disabled:!c||n,className:"run-button",children:[n?e.jsx(_,{size:14,className:"spin"}):e.jsx(at,{size:14,fill:"white"}),e.jsx("span",{children:"Run"})]})]})]}),e.jsxs("div",{className:"ide-main",children:[M&&e.jsxs("div",{className:"side-panel left-panel",children:[e.jsxs("div",{className:"panel-header",children:[e.jsx(Q,{size:14}),e.jsx("span",{children:"DataFrame Viewer"})]}),e.jsxs("div",{className:"panel-content dataframe-panel",children:[e.jsxs("div",{className:"df-info",children:[k.length," rows × ",Object.keys(k[0]||{}).length," cols"]}),e.jsx("div",{className:"dataframe-container",children:e.jsxs("table",{className:"dataframe",children:[e.jsx("thead",{children:e.jsxs("tr",{children:[e.jsx("th",{children:"#"}),Object.keys(k[0]||{}).map(o=>e.jsx("th",{children:o},o))]})}),e.jsx("tbody",{children:k.map((o,l)=>e.jsxs("tr",{children:[e.jsx("td",{className:"row-idx",children:l}),Object.entries(o).map(([p,u])=>e.jsx("td",{className:(v==null?void 0:v.row)===l&&(v==null?void 0:v.col)===p?"selected":"",onClick:()=>fe({row:l,col:p}),onDoubleClick:()=>{const C=prompt(`Edit ${p}:`,String(u));C!==null&&we(l,p,C)},children:typeof u=="number"?u.toLocaleString():u},p))]},l))})]})}),e.jsxs("div",{className:"df-actions",children:[e.jsx("button",{onClick:()=>t(o=>o+`
print(df.describe())`),children:"describe()"}),e.jsx("button",{onClick:()=>t(o=>o+`
print(df.info())`),children:"info()"}),e.jsx("button",{onClick:()=>t(o=>o+`
print(df.head())`),children:"head()"})]})]})]}),w&&e.jsxs("div",{className:"side-panel left-panel r-panel",children:[e.jsxs("div",{className:"panel-header",children:[e.jsx(H,{size:14}),e.jsx("span",{children:"R Console"}),m?e.jsx("span",{className:"ready-dot",children:"●"}):e.jsx(_,{size:12,className:"spin"})]}),e.jsxs("div",{className:"panel-content",children:[e.jsxs("div",{className:"r-section",children:[e.jsx("h4",{children:"📊 Run R Code"}),e.jsx("button",{className:"r-btn",onClick:()=>N("x <- c(1, 2, 3, 4, 5); mean(x)"),children:"mean(x)"}),e.jsx("button",{className:"r-btn",onClick:()=>N("x <- c(1, 2, 3, 4, 5); sd(x)"),children:"sd(x)"}),e.jsx("button",{className:"r-btn",onClick:()=>N("x <- c(1, 2, 3, 4, 5); summary(x)"),children:"summary(x)"})]}),e.jsxs("div",{className:"r-section",children:[e.jsx("h4",{children:"📈 Statistical Tests"}),e.jsx("button",{className:"r-btn",onClick:()=>N("x <- rnorm(100); y <- rnorm(100); t.test(x, y)"),children:"t.test()"}),e.jsx("button",{className:"r-btn",onClick:()=>N("x <- c(1,2,3,4,5); y <- c(2,3,5,7,11); cor(x, y)"),children:"cor()"}),e.jsx("button",{className:"r-btn",onClick:()=>N("x <- c(1,2,3,4,5); y <- c(2,4,5,4,5); lm(y ~ x)"),children:"lm()"})]}),e.jsxs("div",{className:"r-section",children:[e.jsx("h4",{children:"🔧 Insert R Template"}),e.jsx("button",{className:"r-btn",onClick:()=>t(`# R Statistical Analysis
x <- c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
y <- c(2.1, 4.0, 5.8, 8.2, 10.1, 12.0, 14.2, 15.9, 18.1, 20.0)

# Linear regression
model <- lm(y ~ x)
summary(model)

# Correlation
cor(x, y)
`),children:"Linear Regression Template"})]})]})]}),e.jsxs("div",{className:"center-panel",children:[e.jsxs("div",{className:"editor-area",children:[e.jsxs("div",{className:"editor-panel",children:[e.jsx("div",{className:"panel-header",children:e.jsx("span",{className:"file-tab active",children:"🐍 script.py"})}),e.jsx("div",{className:"editor-content",children:e.jsx(ht,{initialValue:s,onChange:o=>t(o||"")})})]}),P&&e.jsxs("div",{className:"architecture-panel",children:[e.jsxs("div",{className:"panel-header",children:[e.jsx(G,{size:14}),e.jsx("span",{children:"PyTorch Layers"})]}),e.jsx("div",{className:"architecture-content",children:e.jsxs("div",{className:"layer-palette",children:[e.jsxs("div",{className:"layer-group",children:[e.jsx("h5",{children:"Linear"}),e.jsx("button",{className:"layer-btn linear",onClick:()=>j("self.fc = nn.Linear(in_features, out_features)"),children:"Linear"})]}),e.jsxs("div",{className:"layer-group",children:[e.jsx("h5",{children:"Conv"}),e.jsx("button",{className:"layer-btn conv",onClick:()=>j("self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)"),children:"Conv2d"})]}),e.jsxs("div",{className:"layer-group",children:[e.jsx("h5",{children:"Pool"}),e.jsx("button",{className:"layer-btn pool",onClick:()=>j("self.pool = nn.MaxPool2d(2, 2)"),children:"MaxPool"})]}),e.jsxs("div",{className:"layer-group",children:[e.jsx("h5",{children:"Norm"}),e.jsx("button",{className:"layer-btn norm",onClick:()=>j("self.bn = nn.BatchNorm2d(num_features)"),children:"BatchNorm"})]}),e.jsxs("div",{className:"layer-group",children:[e.jsx("h5",{children:"RNN"}),e.jsx("button",{className:"layer-btn rnn",onClick:()=>j("self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)"),children:"LSTM"}),e.jsx("button",{className:"layer-btn rnn",onClick:()=>j("self.gru = nn.GRU(input_size, hidden_size)"),children:"GRU"})]}),e.jsxs("div",{className:"layer-group",children:[e.jsx("h5",{children:"Activation"}),e.jsx("button",{className:"layer-btn act",onClick:()=>j("x = torch.relu(x)"),children:"ReLU"}),e.jsx("button",{className:"layer-btn act",onClick:()=>j("x = torch.softmax(x, dim=1)"),children:"Softmax"})]})]})})]})]}),e.jsxs("div",{className:"output-panel",children:[e.jsxs("div",{className:"panel-header",children:[e.jsx("span",{className:"panel-tab active",children:"Output"}),e.jsxs("div",{className:"panel-actions",children:[e.jsx("button",{onClick:ge,className:"panel-action-btn",children:D?e.jsx(Ye,{size:14}):e.jsx(Xe,{size:14})}),e.jsx("button",{onClick:je,className:"panel-action-btn",children:e.jsx(it,{size:14})})]})]}),e.jsx("div",{className:"output-content",children:e.jsx("pre",{className:"output-text",children:r||"Click Run to execute"})})]})]}),A&&e.jsxs("div",{className:"side-panel right-panel database-panel",children:[e.jsxs("div",{className:"panel-header",children:[e.jsx(Y,{size:14}),e.jsx("span",{children:"Dataset Browser"}),e.jsx("button",{className:"icon-btn",onClick:B,children:e.jsx(rt,{size:12})})]}),e.jsxs("div",{className:"panel-content",children:[e.jsxs("div",{className:"search-box",children:[e.jsx(lt,{size:14}),e.jsx("input",{type:"text",placeholder:"Search datasets...",value:O,onChange:o=>pe(o.target.value)})]}),e.jsx("div",{className:"filter-tabs",children:["all","kaggle","pytorch","tensorflow","huggingface"].map(o=>e.jsx("button",{className:`filter-tab ${E===o?"active":""}`,onClick:()=>ue(o),children:o==="all"?"All":o.charAt(0).toUpperCase()+o.slice(1)},o))}),e.jsx("div",{className:"dataset-list",children:me?e.jsxs("div",{className:"loading",children:[e.jsx(_,{size:16,className:"spin"})," Loading..."]}):W.slice(0,20).map(o=>e.jsxs("button",{className:"dataset-btn",onClick:()=>t(ve(o)),children:[e.jsx("span",{className:"ds-name",children:o.name}),e.jsxs("span",{className:"ds-meta",children:[o.source," • ",o.category||"Dataset"]})]},o.id+o.source))}),e.jsxs("div",{className:"dataset-count",children:[W.length," datasets available"]})]})]})]}),e.jsxs("footer",{className:"ide-footer",children:[e.jsx("span",{children:"Python 3.11"}),e.jsx("span",{className:"footer-separator",children:"|"}),e.jsxs("span",{children:["R ",m?"4.3":"(click R to load)"]}),e.jsx("span",{className:"footer-separator",children:"|"}),e.jsx("span",{children:"PyTorch"}),e.jsx("div",{className:"footer-right",children:e.jsx("span",{style:{color:c?"#4ade80":"#fbbf24"},children:c?"● Ready":"○ Loading..."})})]})]})}F.createRoot(document.getElementById("root")).render(e.jsx(ke.StrictMode,{children:e.jsx(ut,{})}));
