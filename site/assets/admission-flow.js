(function(){const c=document.getElementById('flow');if(!c)return;const x=c.getContext('2d');
let W,H,dpr=Math.min(devicePixelRatio||1,2);const reduce=matchMedia('(prefers-reduced-motion:reduce)').matches;
const ind='#5B5BF6';function size(){W=c.clientWidth;H=c.height;c.width=W*dpr;c.height=H*dpr;x.setTransform(dpr,0,0,dpr,0,0);}
addEventListener('resize',size);size();
const cx=()=>W*0.5,cy=()=>H*0.5;const inLanes=[.28,.5,.72],outLanes=[.28,.5,.72];
let dots=[];function spawn(){const l=inLanes[Math.floor(Math.random()*3)];dots.push({p:0,y:l,o:outLanes[Math.floor(Math.random()*3)]});}
function frame(t){x.clearRect(0,0,W,H);
 // paths
 x.strokeStyle=getComputedStyle(document.documentElement).getPropertyValue('--line');x.lineWidth=1;
 inLanes.forEach(l=>{x.setLineDash([4,5]);x.beginPath();x.moveTo(0,H*l);x.lineTo(cx(),cy());x.stroke();});
 outLanes.forEach(l=>{x.beginPath();x.moveTo(cx(),cy());x.lineTo(W,H*l);x.stroke();});x.setLineDash([]);
 // aperture
 const pulse=6+3*Math.sin(t/400);x.fillStyle=ind;x.globalAlpha=.15;x.beginPath();x.arc(cx(),cy(),22+pulse,0,7);x.fill();
 x.globalAlpha=1;x.strokeStyle=ind;x.lineWidth=2.5;x.strokeRect(cx()-10,cy()-16,20,32);
 // dots
 dots.forEach(d=>{d.p+=0.012;let px,py;if(d.p<.5){const k=d.p/.5;px=k*cx();py=H*d.y+(cy()-H*d.y)*k;}else{const k=(d.p-.5)/.5;px=cx()+k*(W-cx());py=cy()+(H*d.o-cy())*k;}
   x.fillStyle=ind;x.beginPath();x.arc(px,py,3,0,7);x.fill();});
 dots=dots.filter(d=>d.p<1);requestAnimationFrame(frame);}
if(reduce){spawn();spawn();spawn();frame(0);}else{setInterval(spawn,520);requestAnimationFrame(frame);}
})();
