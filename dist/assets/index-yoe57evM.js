import{r as d,_ as $e,F as Me,R as De}from"./monaco-BePFVKFP.js";import{r as Pe}from"./vendor-CQMrbrXp.js";(function(){const s=document.createElement("link").relList;if(s&&s.supports&&s.supports("modulepreload"))return;for(const a of document.querySelectorAll('link[rel="modulepreload"]'))n(a);new MutationObserver(a=>{for(const i of a)if(i.type==="childList")for(const u of i.addedNodes)u.tagName==="LINK"&&u.rel==="modulepreload"&&n(u)}).observe(document,{childList:!0,subtree:!0});function r(a){const i={};return a.integrity&&(i.integrity=a.integrity),a.referrerPolicy&&(i.referrerPolicy=a.referrerPolicy),a.crossOrigin==="use-credentials"?i.credentials="include":a.crossOrigin==="anonymous"?i.credentials="omit":i.credentials="same-origin",i}function n(a){if(a.ep)return;a.ep=!0;const i=r(a);fetch(a.href,i)}})();var ee={exports:{}},S={};/**
 * @license React
 * react-jsx-runtime.production.min.js
 *
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */var Ee=d,Oe=Symbol.for("react.element"),Ae=Symbol.for("react.fragment"),ze=Object.prototype.hasOwnProperty,Fe=Ee.__SECRET_INTERNALS_DO_NOT_USE_OR_YOU_WILL_BE_FIRED.ReactCurrentOwner,Ie={key:!0,ref:!0,__self:!0,__source:!0};function te(t,s,r){var n,a={},i=null,u=null;r!==void 0&&(i=""+r),s.key!==void 0&&(i=""+s.key),s.ref!==void 0&&(u=s.ref);for(n in s)ze.call(s,n)&&!Ie.hasOwnProperty(n)&&(a[n]=s[n]);if(t&&t.defaultProps)for(n in s=t.defaultProps,s)a[n]===void 0&&(a[n]=s[n]);return{$$typeof:Oe,type:t,key:i,ref:u,props:a,_owner:Fe.current}}S.Fragment=Ae;S.jsx=te;S.jsxs=te;ee.exports=S;var e=ee.exports,F={},U=Pe;F.createRoot=U.createRoot,F.hydrateRoot=U.hydrateRoot;let v=null,C=null,h=null;const Te={},T=async()=>{if(!v)return C||(C=(async()=>{const t=document.createElement("script");t.src="https://cdn.jsdelivr.net/pyodide/v0.24.1/full/pyodide.js",document.head.appendChild(t),await new Promise((s,r)=>{t.onload=()=>s(),t.onerror=()=>r(new Error("Failed to load Pyodide"))}),v=await window.loadPyodide({indexURL:"https://cdn.jsdelivr.net/pyodide/v0.24.1/full/"}),await v.runPythonAsync(`
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
`)})(),C)};let j=null,_=null;const se=async()=>{if(!j)return _||(_=(async()=>{const{WebR:t}=await $e(async()=>{const{WebR:s}=await import("https://webr.r-wasm.org/latest/webr.mjs");return{WebR:s}},[],import.meta.url);j=new t,await j.init()})(),_)},ne=t=>{const s=t.trim();return s?s.startsWith("#")?"comment":/<-/.test(t)||/^\s*library\s*\(/.test(t)||/^\s*c\s*\(/.test(t)||/\$\w+/.test(t)||/^\s*(cat|print)\s*\(.*\\n/.test(t)?"r":(/^\s*(import|from)\s+/.test(t)||/^\s*def\s+\w+\s*\(/.test(t)||/^\s*class\s+\w+/.test(t)||/print\s*\(f?["']/.test(t)||/\[.*for.*in.*\]/.test(t)||/:\s*$/.test(t),"python"):"empty"},Ve=t=>{const s=t.split(`
`),r=[];let n=null,a=[],i=1;for(let u=0;u<s.length;u++){const g=s[u],y=g.trim().toLowerCase();if(y==="#%% python"||y==="# python"){a.length>0&&n&&r.push({lang:n,code:a.join(`
`),startLine:i}),n="python",a=[],i=u+2;continue}if(y==="#%% r"||y==="# r"){a.length>0&&n&&r.push({lang:n,code:a.join(`
`),startLine:i}),n="r",a=[],i=u+2;continue}const x=ne(g);x!=="comment"&&x!=="empty"&&(n===null?(n=x,i=u+1):x!==n&&(a.length>0&&r.push({lang:n,code:a.join(`
`),startLine:i}),n=x,a=[],i=u+1)),a.push(g)}if(a.length>0&&n){const u=a.join(`
`).trim();u&&r.push({lang:n,code:u,startLine:i})}return r},qe=async()=>{if(!(!v||!j))try{const t=await v.runPythonAsync(`
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
        `);if(t&&t!=="{}"){const s=JSON.parse(t);for(const[r,n]of Object.entries(s))Te[r]=n,typeof n=="number"?await j.evalR(`${r} <- ${n}`):Array.isArray(n)?await j.evalR(`${r} <- c(${n.join(", ")})`):typeof n=="string"&&await j.evalR(`${r} <- "${n}"`)}}catch{}},We=async t=>{var s;await T();try{await v.runPythonAsync(t)}catch(r){h==null||h(`Error: ${((s=r.message)==null?void 0:s.split(`
`).pop())||r}
`)}},ae=async t=>{var s;await se(),await qe();try{const r=`capture.output({ ${t} }, type = "output")`,n=await j.evalR(r);try{const a=await n.toArray();if(a&&a.length>0){const i=a.filter(u=>u!=="").join(`
`);i&&(h==null||h(i+`
`))}}catch{const a=await n.toString();a&&(h==null||h(a+`
`))}}catch(r){h==null||h(`R: ${((s=r.message)==null?void 0:s.split(`
`).slice(-2).join(" "))||r}
`)}},J=async t=>{const s=Ve(t);for(const r of s)r.lang==="python"?await We(r.code):await ae(r.code)},Be=t=>{const s=t.split(`
`),r=[];let n="py";for(let a=0;a<s.length;a++){const i=s[a].trim().toLowerCase();if(i==="#%% python"||i==="# python")n="py";else if(i==="#%% r"||i==="# r")n="R";else{const u=ne(s[a]);u==="r"?n="R":u==="python"&&(n="py")}s[a].trim()&&!s[a].trim().startsWith("#")&&r.push({line:a+1,lang:n})}return r},re="ai-ide-files";let R=[];const He=[{name:"script.py",isDirectory:!1,path:"/project/script.py",content:`# Mixed Python + R Code
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
`}],Ue=()=>{try{const t=localStorage.getItem(re);if(t)return JSON.parse(t)}catch{}return[...He]},Je=()=>{try{localStorage.setItem(re,JSON.stringify(R))}catch{}};R=Ue();const I=t=>R.find(s=>s.path===t||s.path==="/project/"+t||s.name===t),Ze={selectDirectory:async()=>"/project",readDir:async t=>R.filter(s=>s.path.startsWith(t)).map(s=>({name:s.name,isDirectory:s.isDirectory,path:s.path})),readFile:async t=>{var s;return((s=I(t))==null?void 0:s.content)||""},saveFile:async(t,s)=>{const r=I(t);return r?r.content=s:R.push({name:t.split("/").pop()||t,isDirectory:!1,path:"/project/"+t,content:s}),Je(),{success:!0}}},Ke={init:()=>{T().catch(console.error)},send:async t=>{const s=t.trim();if(s.startsWith("python ")||s.includes(".py")){const r=s.match(/python\s+["']?([^"'\s]+)["']?/);if(r){const n=I(r[1]);n!=null&&n.content?await J(n.content):h==null||h(`File not found
`)}}else s.length>0&&await J(s)},onData:t=>(h=t,()=>{h=null})},Ye={search:async t=>{try{const r=await(await fetch("./datasets.json")).json();return{success:!0,data:[...r.kaggle.map(a=>({...a,source:"kaggle"})),...r.tensorflow.map(a=>({...a,source:"tensorflow"})),...r.pytorch.map(a=>({...a,source:"pytorch"})),...r.huggingface.map(a=>({...a,source:"huggingface"}))].filter(a=>a.name.toLowerCase().includes(t.toLowerCase())||a.id.toLowerCase().includes(t.toLowerCase())).slice(0,20)}}catch{return{success:!1,data:[]}}}};window.fileSystem=Ze;window.terminal=Ke;window.appControl={setMode:()=>{}};window.kaggle=Ye;window.analysis={checkDeps:async()=>({missing:[]}),recommend:async()=>({success:!0,recommendation:""})};window.pyodideReady=T();window.runR=ae;window.loadWebR=se;window.getLanguageAnnotations=Be;/**
 * @license lucide-react v0.562.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const Ge=t=>t.replace(/([a-z0-9])([A-Z])/g,"$1-$2").toLowerCase(),Qe=t=>t.replace(/^([A-Z])|[\s-_]+(\w)/g,(s,r,n)=>n?n.toUpperCase():r.toLowerCase()),Z=t=>{const s=Qe(t);return s.charAt(0).toUpperCase()+s.slice(1)},oe=(...t)=>t.filter((s,r,n)=>!!s&&s.trim()!==""&&n.indexOf(s)===r).join(" ").trim(),Xe=t=>{for(const s in t)if(s.startsWith("aria-")||s==="role"||s==="title")return!0};/**
 * @license lucide-react v0.562.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */var et={xmlns:"http://www.w3.org/2000/svg",width:24,height:24,viewBox:"0 0 24 24",fill:"none",stroke:"currentColor",strokeWidth:2,strokeLinecap:"round",strokeLinejoin:"round"};/**
 * @license lucide-react v0.562.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const tt=d.forwardRef(({color:t="currentColor",size:s=24,strokeWidth:r=2,absoluteStrokeWidth:n,className:a="",children:i,iconNode:u,...g},y)=>d.createElement("svg",{ref:y,...et,width:s,height:s,stroke:t,strokeWidth:n?Number(r)*24/Number(s):r,className:oe("lucide",a),...!i&&!Xe(g)&&{"aria-hidden":"true"},...g},[...u.map(([x,L])=>d.createElement(x,L)),...Array.isArray(i)?i:[i]]));/**
 * @license lucide-react v0.562.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const m=(t,s)=>{const r=d.forwardRef(({className:n,...a},i)=>d.createElement(tt,{ref:i,iconNode:s,className:oe(`lucide-${Ge(Z(t))}`,`lucide-${t}`,n),...a}));return r.displayName=Z(t),r};/**
 * @license lucide-react v0.562.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const st=[["path",{d:"M2.97 12.92A2 2 0 0 0 2 14.63v3.24a2 2 0 0 0 .97 1.71l3 1.8a2 2 0 0 0 2.06 0L12 19v-5.5l-5-3-4.03 2.42Z",key:"lc1i9w"}],["path",{d:"m7 16.5-4.74-2.85",key:"1o9zyk"}],["path",{d:"m7 16.5 5-3",key:"va8pkn"}],["path",{d:"M7 16.5v5.17",key:"jnp8gn"}],["path",{d:"M12 13.5V19l3.97 2.38a2 2 0 0 0 2.06 0l3-1.8a2 2 0 0 0 .97-1.71v-3.24a2 2 0 0 0-.97-1.71L17 10.5l-5 3Z",key:"8zsnat"}],["path",{d:"m17 16.5-5-3",key:"8arw3v"}],["path",{d:"m17 16.5 4.74-2.85",key:"8rfmw"}],["path",{d:"M17 16.5v5.17",key:"k6z78m"}],["path",{d:"M7.97 4.42A2 2 0 0 0 7 6.13v4.37l5 3 5-3V6.13a2 2 0 0 0-.97-1.71l-3-1.8a2 2 0 0 0-2.06 0l-3 1.8Z",key:"1xygjf"}],["path",{d:"M12 8 7.26 5.15",key:"1vbdud"}],["path",{d:"m12 8 4.74-2.85",key:"3rx089"}],["path",{d:"M12 13.5V8",key:"1io7kd"}]],K=m("boxes",st);/**
 * @license lucide-react v0.562.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const nt=[["path",{d:"M5 21v-6",key:"1hz6c0"}],["path",{d:"M12 21V3",key:"1lcnhd"}],["path",{d:"M19 21V9",key:"unv183"}]],Y=m("chart-no-axes-column",nt);/**
 * @license lucide-react v0.562.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const at=[["path",{d:"M20 6 9 17l-5-5",key:"1gmf2c"}]],rt=m("check",at);/**
 * @license lucide-react v0.562.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const ot=[["rect",{width:"14",height:"14",x:"8",y:"8",rx:"2",ry:"2",key:"17jyea"}],["path",{d:"M4 16c-1.1 0-2-.9-2-2V4c0-1.1.9-2 2-2h10c1.1 0 2 .9 2 2",key:"zix9uf"}]],it=m("copy",ot);/**
 * @license lucide-react v0.562.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const ct=[["ellipse",{cx:"12",cy:"5",rx:"9",ry:"3",key:"msslwz"}],["path",{d:"M3 5V19A9 3 0 0 0 21 19V5",key:"1wlel7"}],["path",{d:"M3 12A9 3 0 0 0 21 12",key:"mv7ke4"}]],G=m("database",ct);/**
 * @license lucide-react v0.562.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const lt=[["circle",{cx:"12",cy:"9",r:"1",key:"124mty"}],["circle",{cx:"19",cy:"9",r:"1",key:"1ruzo2"}],["circle",{cx:"5",cy:"9",r:"1",key:"1a8b28"}],["circle",{cx:"12",cy:"15",r:"1",key:"1e56xg"}],["circle",{cx:"19",cy:"15",r:"1",key:"1a92ep"}],["circle",{cx:"5",cy:"15",r:"1",key:"5r1jwy"}]],dt=m("grip-horizontal",lt);/**
 * @license lucide-react v0.562.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const ut=[["path",{d:"M21 12a9 9 0 1 1-6.219-8.56",key:"13zald"}]],Q=m("loader-circle",ut);/**
 * @license lucide-react v0.562.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const ht=[["path",{d:"M5 5a2 2 0 0 1 3.008-1.728l11.997 6.998a2 2 0 0 1 .003 3.458l-12 7A2 2 0 0 1 5 19z",key:"10ikf1"}]],pt=m("play",ht);/**
 * @license lucide-react v0.562.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const mt=[["path",{d:"M3 12a9 9 0 0 1 9-9 9.75 9.75 0 0 1 6.74 2.74L21 8",key:"v9h5vc"}],["path",{d:"M21 3v5h-5",key:"1q7to0"}],["path",{d:"M21 12a9 9 0 0 1-9 9 9.75 9.75 0 0 1-6.74-2.74L3 16",key:"3uifl3"}],["path",{d:"M8 16H3v5",key:"1cv678"}]],yt=m("refresh-cw",mt);/**
 * @license lucide-react v0.562.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const ft=[["path",{d:"M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8",key:"1357e3"}],["path",{d:"M3 3v5h5",key:"1xhq8a"}]],xt=m("rotate-ccw",ft);/**
 * @license lucide-react v0.562.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const gt=[["path",{d:"m21 21-4.34-4.34",key:"14j7rj"}],["circle",{cx:"11",cy:"11",r:"8",key:"4ej97u"}]],jt=m("search",gt);/**
 * @license lucide-react v0.562.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const wt=[["path",{d:"M9 3H5a2 2 0 0 0-2 2v4m6-6h10a2 2 0 0 1 2 2v4M9 3v18m0 0h10a2 2 0 0 0 2-2V9M9 21H5a2 2 0 0 1-2-2V9m0 0h18",key:"gugj83"}]],X=m("table-2",wt),vt=({initialValue:t="# Write your code here",language:s="python",onChange:r,theme:n="vs-dark"})=>e.jsx("div",{style:{width:"100%",height:"100%",overflow:"hidden"},children:e.jsx(Me,{height:"100%",defaultLanguage:s,defaultValue:t,theme:n,onChange:r,options:{minimap:{enabled:!1},fontSize:14,scrollBeyondLastLine:!1,automaticLayout:!0,padding:{top:16},fontFamily:"'JetBrains Mono', 'Fira Code', Consolas, monospace"}})}),Nt=[{id:1,name:"Alice",age:28,salary:55e3,dept:"Engineering"},{id:2,name:"Bob",age:34,salary:62e3,dept:"Marketing"},{id:3,name:"Charlie",age:25,salary:48e3,dept:"Engineering"},{id:4,name:"Diana",age:31,salary:71e3,dept:"Sales"},{id:5,name:"Eve",age:29,salary:58e3,dept:"Engineering"}],bt=t=>{const s=t.trim();return!s||s.startsWith("#")?null:/<-/.test(t)||/^\s*library\s*\(/.test(t)||/^\s*c\s*\(/.test(t)?"R":(/print\s*\(f?["']/.test(t)||/^\s*(import|from|def|class)\s+/.test(t)||/\[.*for.*in.*\]/.test(t),"py")};function kt(){const[t,s]=d.useState(`# Python + R Notebook
# Variables automatically shared!

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
`),[r,n]=d.useState(""),[a,i]=d.useState(!1),[u,g]=d.useState(!1),[y,x]=d.useState(!1),[L,V]=d.useState(!1),[$,ie]=d.useState(!1),[M,ce]=d.useState(!1),[D,le]=d.useState(!1),[P,de]=d.useState(!1),[ue,he]=d.useState(200),pe=d.useRef(null),E=d.useRef(!1),[me,ye]=d.useState([]),[q,fe]=d.useState(""),[O,xe]=d.useState("all"),[ge,W]=d.useState(!1),[N,je]=d.useState(Nt),[w,we]=d.useState(null),[ve,Ne]=d.useState([]);d.useEffect(()=>{const o=t.split(`
`),c=[];let l="py";for(let p=0;p<o.length;p++){const f=o[p].trim().toLowerCase();if(f==="#%% python"||f==="# python")l="py";else if(f==="#%% r"||f==="# r")l="R";else{const H=bt(o[p]);H&&(l=H)}o[p].trim()&&!o[p].trim().startsWith("#")&&c.push({line:p+1,lang:l})}Ne(c)},[t]);const B=async()=>{W(!0);try{const c=await(await fetch("./datasets.json")).json();ye([...c.kaggle.map(l=>({...l,source:"kaggle"})),...c.tensorflow.map(l=>({...l,source:"tensorflow"})),...c.pytorch.map(l=>({...l,source:"pytorch"})),...c.huggingface.map(l=>({...l,source:"huggingface"}))])}catch(o){console.error(o)}W(!1)};d.useEffect(()=>{B(),window.pyodideReady&&window.pyodideReady.then(()=>{g(!0),n(`✓ Python ready
`)}).catch(c=>n(`Error: ${c}
`)),window.loadWebR&&window.loadWebR().then(()=>{x(!0),n(c=>c+`✓ R ready
`)}).catch(console.error);const o=window.terminal.onData(c=>{const l=c.replace(/\r\n/g,`
`).replace(/\r/g,`
`);(l.trim()||l.includes(`
`))&&n(p=>p+l)});return window.terminal.init(),o},[]);const be=d.useCallback(()=>{E.current=!0,document.body.style.cursor="row-resize",document.body.style.userSelect="none"},[]),A=d.useCallback(o=>{if(!E.current)return;const c=document.querySelector(".center-panel");if(!c)return;const p=c.getBoundingClientRect().bottom-o.clientY;he(Math.max(100,Math.min(500,p)))},[]),z=d.useCallback(()=>{E.current=!1,document.body.style.cursor="",document.body.style.userSelect=""},[]);d.useEffect(()=>(document.addEventListener("mousemove",A),document.addEventListener("mouseup",z),()=>{document.removeEventListener("mousemove",A),document.removeEventListener("mouseup",z)}),[A,z]);const ke=async()=>{!u||a||(i(!0),n(""),await window.fileSystem.saveFile("script.py",t),window.terminal.send(`python "script.py"
`),setTimeout(()=>i(!1),800))},b=async o=>{if(window.runR){n(c=>c+`
`);try{await window.runR(o)}catch(c){n(l=>l+`R: ${c.message}
`)}}},Re=()=>n(""),Ce=()=>{navigator.clipboard.writeText(r),V(!0),setTimeout(()=>V(!1),2e3)},_e=(o,c,l)=>{const p=[...N],f=parseFloat(l);p[o][c]=isNaN(f)?l:f,je(p)},k=o=>s(c=>c+`
${o}`),Se=me.filter(o=>{const c=o.name.toLowerCase().includes(q.toLowerCase()),l=O==="all"||o.source===O;return c&&l}),Le=o=>o.source==="pytorch"?`# ${o.name}
from torchvision import datasets
train = datasets.${o.id}(root='./data', train=True, download=True)
print(f"Samples: {len(train)}")`:o.source==="tensorflow"?`# ${o.name}
import tensorflow as tf
(x_train, y_train), _ = tf.keras.datasets.${o.id}.load_data()
print(f"Shape: {x_train.shape}")`:`# ${o.name} from ${o.source}
# ID: ${o.id}
print("Dataset: ${o.name}")`;return e.jsxs("div",{className:"ide-container",children:[e.jsxs("header",{className:"ide-header",children:[e.jsxs("div",{className:"header-left",children:[e.jsx("span",{className:"app-title",children:"🔥 AI IDE"}),e.jsx("span",{className:"badge pytorch",children:"Py"}),e.jsx("span",{className:"badge r-badge",children:"R"})]}),e.jsxs("div",{className:"header-center",children:[e.jsxs("button",{className:`toggle-btn ${$?"active":""}`,onClick:()=>ie(!$),children:[e.jsx(K,{size:14}),e.jsx("span",{children:"Arch"})]}),e.jsxs("button",{className:`toggle-btn ${M?"active":""}`,onClick:()=>ce(!M),children:[e.jsx(G,{size:14}),e.jsx("span",{children:"Data"})]}),e.jsxs("button",{className:`toggle-btn ${D?"active":""}`,onClick:()=>le(!D),children:[e.jsx(X,{size:14}),e.jsx("span",{children:"DF"})]}),e.jsxs("button",{className:`toggle-btn ${P?"active":""}`,onClick:()=>de(!P),children:[e.jsx(Y,{size:14}),e.jsx("span",{children:"R"})]})]}),e.jsxs("div",{className:"header-right",children:[!u&&e.jsx("span",{className:"status-dot loading",children:"Py"}),!y&&e.jsx("span",{className:"status-dot loading",children:"R"}),u&&e.jsx("span",{className:"status-dot ready",children:"Py"}),y&&e.jsx("span",{className:"status-dot ready",children:"R"}),e.jsxs("button",{onClick:ke,disabled:!u||a,className:"run-button",children:[a?e.jsx(Q,{size:14,className:"spin"}):e.jsx(pt,{size:14,fill:"white"}),e.jsx("span",{children:"Run"})]})]})]}),e.jsxs("div",{className:"ide-main",children:[D&&e.jsxs("div",{className:"side-panel left-panel",children:[e.jsxs("div",{className:"panel-header",children:[e.jsx(X,{size:14}),e.jsx("span",{children:"DataFrame"})]}),e.jsxs("div",{className:"panel-content dataframe-panel",children:[e.jsxs("div",{className:"df-info",children:[N.length," × ",Object.keys(N[0]||{}).length]}),e.jsx("div",{className:"dataframe-container",children:e.jsxs("table",{className:"dataframe",children:[e.jsx("thead",{children:e.jsxs("tr",{children:[e.jsx("th",{children:"#"}),Object.keys(N[0]||{}).map(o=>e.jsx("th",{children:o},o))]})}),e.jsx("tbody",{children:N.map((o,c)=>e.jsxs("tr",{children:[e.jsx("td",{className:"row-idx",children:c}),Object.entries(o).map(([l,p])=>e.jsx("td",{className:(w==null?void 0:w.row)===c&&(w==null?void 0:w.col)===l?"selected":"",onClick:()=>we({row:c,col:l}),onDoubleClick:()=>{const f=prompt(`${l}:`,String(p));f!==null&&_e(c,l,f)},children:typeof p=="number"?p.toLocaleString():p},l))]},c))})]})}),e.jsxs("div",{className:"df-actions",children:[e.jsx("button",{onClick:()=>s(o=>o+`
print(df.describe())`),children:"describe"}),e.jsx("button",{onClick:()=>s(o=>o+`
print(df.head())`),children:"head"})]})]})]}),P&&e.jsxs("div",{className:"side-panel left-panel r-panel",children:[e.jsxs("div",{className:"panel-header",children:[e.jsx(Y,{size:14}),e.jsx("span",{children:"R"}),y&&e.jsx("span",{className:"ready-dot",children:"●"})]}),e.jsxs("div",{className:"panel-content",children:[e.jsxs("div",{className:"r-section",children:[e.jsx("h4",{children:"Stats"}),e.jsx("button",{className:"r-btn",onClick:()=>b("x <- c(1,2,3,4,5); mean(x)"),children:"mean()"}),e.jsx("button",{className:"r-btn",onClick:()=>b("x <- c(1,2,3,4,5); sd(x)"),children:"sd()"}),e.jsx("button",{className:"r-btn",onClick:()=>b("x <- c(1,2,3,4,5); summary(x)"),children:"summary()"})]}),e.jsxs("div",{className:"r-section",children:[e.jsx("h4",{children:"Tests"}),e.jsx("button",{className:"r-btn",onClick:()=>b("t.test(rnorm(10), rnorm(10))"),children:"t.test()"}),e.jsx("button",{className:"r-btn",onClick:()=>b("cor(c(1,2,3,4,5), c(2,4,5,4,5))"),children:"cor()"})]})]})]}),e.jsxs("div",{className:"center-panel",children:[e.jsxs("div",{className:"editor-area",children:[e.jsxs("div",{className:"editor-panel",children:[e.jsxs("div",{className:"panel-header",children:[e.jsx("span",{className:"file-tab active",children:"📄 script.py"}),e.jsxs("div",{className:"lang-legend",children:[e.jsx("span",{className:"lang-tag py",children:"py"}),e.jsx("span",{className:"lang-tag r",children:"R"})]})]}),e.jsxs("div",{className:"editor-with-gutter",children:[e.jsx("div",{className:"language-gutter",children:t.split(`
`).map((o,c)=>{const l=ve.find(p=>p.line===c+1);return e.jsx("div",{className:"gutter-line",children:l&&e.jsx("span",{className:`gutter-lang ${l.lang}`,children:l.lang})},c)})}),e.jsx("div",{className:"editor-content",children:e.jsx(vt,{initialValue:t,onChange:o=>s(o||"")})})]})]}),$&&e.jsxs("div",{className:"architecture-panel",children:[e.jsxs("div",{className:"panel-header",children:[e.jsx(K,{size:14}),e.jsx("span",{children:"PyTorch"})]}),e.jsx("div",{className:"architecture-content",children:e.jsx("div",{className:"layer-palette",children:e.jsxs("div",{className:"layer-group",children:[e.jsx("button",{className:"layer-btn linear",onClick:()=>k("self.fc = nn.Linear(in, out)"),children:"Linear"}),e.jsx("button",{className:"layer-btn conv",onClick:()=>k("self.conv = nn.Conv2d(in, out, 3)"),children:"Conv2d"}),e.jsx("button",{className:"layer-btn pool",onClick:()=>k("self.pool = nn.MaxPool2d(2)"),children:"MaxPool"}),e.jsx("button",{className:"layer-btn rnn",onClick:()=>k("self.lstm = nn.LSTM(in, hidden)"),children:"LSTM"}),e.jsx("button",{className:"layer-btn act",onClick:()=>k("x = torch.relu(x)"),children:"ReLU"})]})})})]})]}),e.jsx("div",{className:"resize-handle",ref:pe,onMouseDown:be,children:e.jsx(dt,{size:12})}),e.jsxs("div",{className:"output-panel",style:{height:ue},children:[e.jsxs("div",{className:"panel-header",children:[e.jsx("span",{className:"panel-tab",children:"Output"}),e.jsxs("div",{className:"panel-actions",children:[e.jsx("button",{onClick:Ce,className:"panel-action-btn",children:L?e.jsx(rt,{size:12}):e.jsx(it,{size:12})}),e.jsx("button",{onClick:Re,className:"panel-action-btn",children:e.jsx(xt,{size:12})})]})]}),e.jsx("div",{className:"output-content",children:e.jsx("pre",{className:"output-text",children:r||"Click Run to execute"})})]})]}),M&&e.jsxs("div",{className:"side-panel right-panel database-panel",children:[e.jsxs("div",{className:"panel-header",children:[e.jsx(G,{size:14}),e.jsx("span",{children:"Data"}),e.jsx("button",{className:"icon-btn",onClick:B,children:e.jsx(yt,{size:11})})]}),e.jsxs("div",{className:"panel-content",children:[e.jsxs("div",{className:"search-box",children:[e.jsx(jt,{size:12}),e.jsx("input",{placeholder:"Search...",value:q,onChange:o=>fe(o.target.value)})]}),e.jsx("div",{className:"filter-tabs",children:["all","kaggle","pytorch","tensorflow"].map(o=>e.jsx("button",{className:`filter-tab ${O===o?"active":""}`,onClick:()=>xe(o),children:o},o))}),e.jsx("div",{className:"dataset-list",children:ge?e.jsx("div",{className:"loading",children:e.jsx(Q,{size:14,className:"spin"})}):Se.slice(0,15).map(o=>e.jsxs("button",{className:"dataset-btn",onClick:()=>s(Le(o)),children:[e.jsx("span",{className:"ds-name",children:o.name}),e.jsx("span",{className:"ds-meta",children:o.source})]},o.id+o.source))})]})]})]}),e.jsxs("footer",{className:"ide-footer",children:[e.jsxs("span",{children:["Python ",u?"✓":"..."]}),e.jsx("span",{className:"footer-separator",children:"|"}),e.jsxs("span",{children:["R ",y?"✓":"..."]})]})]})}F.createRoot(document.getElementById("root")).render(e.jsx(De.StrictMode,{children:e.jsx(kt,{})}));
