'use strict';
const D=window.LECTURE_DATA,mode=document.body.dataset.mode,$=id=>document.getElementById(id);
const C=$('plot'),ctx=C.getContext('2d'),W=1600,H=530;
const gold='#d99700',blue='#253b53',accent='#087bb9',red='#c74842';
function predict(t,x){return t.feature?predict(x[t.feature-1]<t.cut?t.left:t.right,x):t.value}
function vote(trees,x){return trees.reduce((s,t)=>s+predict(t,x),0)>=0?1:-1}
function boosted(rounds,x){return rounds.reduce((s,r)=>s+r.alpha*(x[r.feature-1]<r.cut?r.polarity:-r.polarity),0)>=0?1:-1}
function accuracy(f,data){return data.y.reduce((n,y,i)=>n+(f(data.X[i])===y),0)/data.y.length}
function impurity(p,kind){return kind==='entropy'?(p<=0||p>=1?0:-p*Math.log(p)-(1-p)*Math.log1p(-p)):p*(1-p)}
function gain(t,kind){const f=+$('fraction')?.value||.5,ns=D.signal.filter(x=>x<t).length/D.signal.length,nb=D.background.filter(x=>x<t).length/D.background.length,L=f*ns+(1-f)*nb,R=1-L;if(L<=0||R<=0)return 0;return impurity(f,kind)-L*impurity(f*ns/L,kind)-R*impurity(f*(1-ns)/R,kind)}
function text(s,x,y,size=23,color=blue){ctx.fillStyle=color;ctx.font=`${size}px "Helvetica Neue",Arial`;ctx.fillText(s,x,y)}
function frame(x,y,w,h,title,xlabel,ylabel,xmax=1,ymax=1){text(title,x,y-20,28);ctx.strokeStyle='#c3d0da';ctx.lineWidth=1;ctx.strokeRect(x,y,w,h);for(let i=0;i<=4;i++){let f=i/4;text((f*xmax).toFixed(xmax<=1?2:0),x+f*w-13,y+h+29,18,'#617484');text((f*ymax).toFixed(ymax<=1?2:1),x-43,y+h-f*h+7,18,'#617484')}text(xlabel,x+w/2-70,y+h+65,22);ctx.save();ctx.translate(x-62,y+h/2+70);ctx.rotate(-Math.PI/2);text(ylabel,0,0,22);ctx.restore();return {x,y,w,h,xmax,ymax}}
function line(p,pts,color,width=3){ctx.strokeStyle=color;ctx.lineWidth=width;ctx.beginPath();pts.forEach(([x,y],i)=>{const a=p.x+x/p.xmax*p.w,b=p.y+p.h-y/p.ymax*p.h;i?ctx.lineTo(a,b):ctx.moveTo(a,b)});ctx.stroke()}
function dot(p,x,y,color,r=6){ctx.beginPath();ctx.arc(p.x+x/p.xmax*p.w,p.y+p.h-y/p.ymax*p.h,r,0,Math.PI*2);ctx.fillStyle=color;ctx.fill();ctx.strokeStyle='white';ctx.lineWidth=1;ctx.stroke()}
function density(x,signal=true){return signal?30*x*(1-x)**4:30*x**4*(1-x)}
function dist(p,t){const pts=Array.from({length:201},(_,i)=>i/200);ctx.fillStyle='#e7f3fc';ctx.fillRect(p.x,p.y,t*p.w,p.h);line(p,pts.map(x=>[x,density(x)]),gold);line(p,pts.map(x=>[x,density(x,false)]),blue);line(p,[[t,0],[t,p.ymax]],accent,2);text('accept x < cut',p.x+12,p.y+28,20,accent)}
function boundary(p,f,data,weights){const n=65;for(let j=0;j<n;j++)for(let i=0;i<n;i++){ctx.fillStyle=f([(i+.5)/n,(j+.5)/n])===1?'#ffe4a3':'#d9e5ef';ctx.fillRect(p.x+i*p.w/n,p.y+(n-j-1)*p.h/n,p.w/n+.5,p.h/n+.5)}for(let i=0;i<data.X.length;i++)dot(p,...data.X[i],data.y[i]===1?gold:blue,weights?Math.min(15,Math.max(2.3,Math.sqrt(weights[i]*data.y.length)*4)):3.3)}
function stat(items){$('stats').innerHTML=items.map(([k,v])=>`<span>${k}<br><b>${v}</b></span>`).join('')}
const pc=x=>(100*x).toFixed(1)+'%';
function draw(){ctx.clearRect(0,0,W,H);document.querySelectorAll('input').forEach(e=>{const o=$(e.id+'-value');if(o)o.value=e.value});
 if(mode==='cuts'){
  const t=+$('cut').value,f=+$('fraction').value,p=frame(90,50,600,350,'A threshold selects both classes','feature x','density',1,3);dist(p,t);
  const q=frame(900,50,600,350,'ROC operating point','signal efficiency','background rejection');
  const es=x=>D.signal.filter(a=>a<x).length/D.signal.length,eb=x=>D.background.filter(a=>a<x).length/D.background.length;
  line(q,Array.from({length:201},(_,i)=>[es(i/200),1-eb(i/200)]),accent);dot(q,es(t),1-eb(t),red,9);
  const sel=f*es(t)+(1-f)*eb(t);stat([['Signal efficiency',pc(es(t))],['Background rejection',pc(1-eb(t))],['Selected purity',sel?pc(f*es(t)/sel):'no selected events'],['Signal fraction before cut',pc(f)]]);
 } else if(mode==='split'){
  const t=+$('cut').value,kind=$('criterion').value,p=frame(90,50,600,350,'Parent population and candidate cut','feature x','density',1,3);dist(p,t);
  const q=frame(900,50,600,350,kind==='gini'?'Gini gain: p(1 − p)':'Entropy gain: natural logarithm','candidate cut','impurity decrease',1,kind==='gini'?.25:.7);
  const vals=Array.from({length:201},(_,i)=>[i/200,gain(i/200,kind)]);line(q,vals,accent);dot(q,t,gain(t,kind),red,9);
  const best=vals.reduce((a,b)=>a[1]>b[1]?a:b);stat([['Gain at selected cut',gain(t,kind).toFixed(4)],['Best cut on 0.005 grid',best[0].toFixed(3)],['Best gain',best[1].toFixed(4)],['Parent impurity',impurity(.5,kind).toFixed(4)]]);
 } else if(mode==='depth'){
  const depth=+$('depth').value,n=+$('trees').value,show=$('sample').value,data=D[show],f=x=>predict(D.trees[depth-1],x),b=x=>vote(D.bag.slice(0,n),x);
  boundary(frame(90,50,600,350,`Single tree · depth ${depth}`,'feature 1','feature 2'),f,data);
  boundary(frame(900,50,600,350,`Bagging · ${n} trees, each depth ≤ 6`,'feature 1','feature 2'),b,data);
  stat([['Tree training accuracy',pc(accuracy(f,D.train))],['Tree validation accuracy',pc(accuracy(f,D.validation))],['Bagging training accuracy',pc(accuracy(b,D.train))],['Bagging validation accuracy',pc(accuracy(b,D.validation))]]);
 } else {
  const m=+$('round').value,rounds=D.boost.slice(0,m),r=rounds.at(-1),f=x=>x[r.feature-1]<r.cut?r.polarity:-r.polarity,b=x=>boosted(rounds,x);
  boundary(frame(90,50,600,350,`Stump ${m} · point area shows incoming weight`,'feature 1','feature 2'),f,D.train,r.before);
  boundary(frame(900,50,600,350,`Weighted vote after ${m} rounds`,'feature 1','feature 2'),b,D.train);
  stat([['Weighted stump error',pc(r.error)],['Stump vote α',r.alpha.toFixed(3)],['Ensemble training accuracy',pc(accuracy(b,D.train))],['Ensemble validation accuracy',pc(accuracy(b,D.validation))]]);
 }
}
document.querySelectorAll('input,select').forEach(e=>e.addEventListener('input',draw));
$('reset').addEventListener('click',()=>{document.querySelectorAll('input').forEach(e=>e.value=e.defaultValue);document.querySelectorAll('select').forEach(e=>e.selectedIndex=0);draw()});
if($('next'))$('next').addEventListener('click',()=>{$('round').value=Math.min(D.boost.length,+$('round').value+1);draw()});
if($('best'))$('best').addEventListener('click',()=>{let t=0,g=-1;for(let i=0;i<=200;i++){let v=gain(i/200,$('criterion').value);if(v>g){g=v;t=i/200}}$('cut').value=t;draw()});
window.LAB={predict,vote,boosted,accuracy,impurity,gain,draw};draw();
