(function(){const f=document.getElementById('dp'),msg=document.getElementById('msg');if(!f)return;
f.addEventListener('submit',async e=>{e.preventDefault();msg.className='formmsg';msg.textContent='Sending…';
 const b={email:f.email.value.trim(),node_count:f.node_count.value,deployment:f.deployment.value,note:f.note.value.trim()};
 if(!/^[^@\s]+@[^@\s]+\.[^@\s]+$/.test(b.email)){msg.className='formmsg err';msg.textContent='Please enter a valid email.';return;}
 try{const r=await fetch('/api/design-partner',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify(b)});
   const d=await r.json();if(d.ok){f.reset();msg.className='formmsg ok';msg.textContent="You're on the list. We'll be in touch.";}
   else{msg.className='formmsg err';msg.textContent='Something went wrong — try again.';}}
 catch(_){msg.className='formmsg err';msg.textContent='Network error — try again.';}});
})();
