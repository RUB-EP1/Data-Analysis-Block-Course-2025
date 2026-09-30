(function (root) {
  'use strict';
  function compute(w = 1, x = 2, t = 5, eta = .01) {
    const a = w*x, b = a*a, c = b+a, loss = .5*(c-t)**2;
    const dc = c-t, db = dc, direct = dc, square = 2*a*db;
    const da = direct+square, dw = da*x, dx = da*w;
    const nextW = w-eta*dw, nextA = nextW*x, nextC = nextA*nextA+nextA;
    return {w,x,t,eta,a,b,c,loss,dc,db,direct,square,da,dw,dx,nextW,nextC,nextLoss:.5*(nextC-t)**2};
  }
  const f = n => Math.abs(n) < 1e-10 ? '0' : Number(n.toPrecision(5)).toString();
  function svgFor(stage, v = compute()) {
    const red = '#d91e0b', blue = '#007dc5';
    const edge = (points, active) => {
      const p = active ? [...points].reverse() : points;
      return `<polyline points="${p.map(x=>x.join(',')).join(' ')}" fill="none" stroke="${active?red:'#bbc2c8'}" stroke-width="${active?5:3}" marker-end="url(#${active?'red':'grey'})"/>`;
    };
    const adj = (key) => {
      if(stage>=6 && key==='w')return f(v.dw);
      if(stage>=6 && key==='x')return f(v.dx);
      if(stage>=5 && key==='a')return f(v.da);
      if(stage===4 && key==='a')return `${f(v.direct)} + ${f(v.square)} ?`;
      if(stage===3 && key==='a')return `${f(v.direct)} (partial)`;
      if(stage>=3 && key==='b')return f(v.db);
      if(stage>=2 && key==='c')return f(v.dc);
      if(stage>=2 && key==='L')return '1';
      return '';
    };
    const nodes=[['w','w',v.w,50,35,180],['x','x',v.x,50,205,180],
      ['a','a = wx',v.a,355,130,220],['b','b = a²',v.b,685,20,220],
      ['c','c = b + a',v.c,1020,130,220],['L','L = ½(c − t)²',v.loss,1370,130,270]];
    const shapes=nodes.map(([key,label,val,x,y,w])=>{
      const av=adj(key), known=stage>=1||key==='w'||key==='x';
      return `<g><rect x="${x}" y="${y}" width="${w}" height="85" rx="19" fill="${av?'#fff5f3':'#f2faff'}" stroke="${av?red:blue}" stroke-width="3"/><text x="${x+w/2}" y="${y+34}" text-anchor="middle" font-size="28" fill="#111">${label}</text><text x="${x+w/2}" y="${y+69}" text-anchor="middle" font-size="29" fill="${blue}">${known?f(val):'?'}</text>${av?`<text x="${x+w/2}" y="${y+115}" text-anchor="middle" font-size="25" fill="${red}">∂L/∂${key} = ${av}</text>`:''}</g>`;
    }).join('');
    return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1700 370" role="img" aria-label="Forward computation values and reverse sensitivities for the worked example"><defs>${['grey','red'].map((name)=>`<marker id="${name}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="${name==='red'?red:'#bbc2c8'}"/></marker>`).join('')}</defs><g font-family="Helvetica Neue,Helvetica,Arial,sans-serif">`
      +edge([[230,77],[295,77],[295,150],[355,150]],stage>=6)
      +edge([[230,247],[295,247],[295,195],[355,195]],stage>=6)
      +edge([[575,150],[630,150],[630,62],[685,62]],stage>=4)
      +edge([[905,62],[970,62],[970,150],[1020,150]],stage>=3)
      +edge([[575,190],[610,190],[610,297],[980,297],[980,190],[1020,190]],stage>=3)
      +edge([[1240,172],[1370,172]],stage>=2)
      +`<text x="1505" y="55" text-anchor="middle" font-size="27" fill="${blue}">target t = ${f(v.t)}</text>`
      +edge([[1505,68],[1505,130]],false)+shapes
      +`<text x="790" y="333" text-anchor="middle" font-size="23" fill="#5e5e5e">direct path from a to c</text><text x="45" y="362" font-size="22" fill="${blue}">Blue: forward values</text><text x="340" y="362" font-size="22" fill="${red}">Red: backward sensitivities</text></g></svg>`;
  }
  function description(stage,v){
    return [
      ['The computation graph','Two paths connect a to the loss. Predict the forward values first.'],
      ['1. Forward pass',`a = ${f(v.a)}, b = ${f(v.b)}, c = ${f(v.c)}, L = ${f(v.loss)}. Keep these values.`],
      ['2. Loss derivative',`Seed ∂L/∂L = 1. Then ∂L/∂c = c − t = ${f(v.dc)}.`],
      ['3. Addition',`c = b + a sends ${f(v.dc)} to each input. The contribution to a is still partial.`],
      ['4. Square',`The path through b contributes 2a × ∂L/∂b = ${f(v.square)}. What is the total at a?`],
      ['5. Accumulation',`∂L/∂a = ${f(v.direct)} + ${f(v.square)} = ${f(v.da)}. Add both paths before continuing.`],
      ['6. Parameter gradient',`∂L/∂w = x × ∂L/∂a = ${f(v.dw)}. Also ∂L/∂x = ${f(v.dx)}.`],
      ['7. Gradient-descent update',`w_new = ${f(v.nextW)}. A fresh forward pass gives L_new = ${f(v.nextLoss)} (${v.nextLoss<v.loss?'decreased':v.nextLoss>v.loss?'increased':'unchanged'} from ${f(v.loss)}).`]
    ][stage];
  }
  const api={compute,svgFor,description,format:f};
  if(typeof module!=='undefined'&&module.exports)module.exports=api;
  root.Backprop=api;
})(typeof window==='undefined'?globalThis:window);
