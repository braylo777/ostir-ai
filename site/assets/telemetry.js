(function(){const seeds={goodput:{v:34,t:88,u:'%'},kv:{v:41,t:94,u:'%'},util:{v:11,t:63,u:'%'}};
const reduce=matchMedia('(prefers-reduced-motion:reduce)').matches;const ind='#5B5BF6';
document.querySelectorAll('.spark').forEach(cv=>{const m=cv.dataset.metric,s=seeds[m];s.hist=Array.from({length:40},(_,i)=>s.v+(s.t-s.v)*(i/40)+(Math.sin(i)*2));
 const x=cv.getContext('2d');function draw(){const W=cv.clientWidth,H=cv.height,dpr=Math.min(devicePixelRatio||1,2);cv.width=W*dpr;cv.height=H*dpr;x.setTransform(dpr,0,0,dpr,0,0);x.clearRect(0,0,W,H);
   const mn=Math.min(...s.hist),mx=Math.max(...s.hist)||1;x.strokeStyle=ind;x.lineWidth=2;x.beginPath();
   s.hist.forEach((v,i)=>{const px=i/(s.hist.length-1)*W,py=H-((v-mn)/(mx-mn||1))*(H-8)-4;i?x.lineTo(px,py):x.moveTo(px,py);});x.stroke();
   x.globalAlpha=.12;x.lineTo(W,H);x.lineTo(0,H);x.closePath();x.fillStyle=ind;x.fill();x.globalAlpha=1;}
 cv.__draw=draw;draw();const out=document.querySelector('[data-out='+m+']');function tick(){s.v+=(s.t-s.v)*0.04+(Math.random()-.5);s.hist.push(s.v);s.hist.shift();out.textContent=Math.round(s.v)+s.u;draw();}
 out.textContent=Math.round(s.v)+s.u;if(!reduce){setInterval(()=>{if(!document.hidden)tick();},900);} });
})();
