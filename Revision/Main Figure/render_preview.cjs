// Optional PNG previews. SVG generation only needs Julia.
// Requires Node.js and sharp; use the Codex bundled runtime or an installed sharp.
const fs = require('node:fs');
const path = require('node:path');
const sharp = require('sharp');
const escape = text => text.replaceAll('&', '&amp;').replaceAll('<', '&lt;');

async function contactSheet(assets, filename, title, rows) {
  const width = 1600, tileWidth = 754;
  let y = 100;
  const layers = [];
  const texts = [
    '<text x="32" y="43" font-size="30" font-weight="600">' + escape(title) + '</text>',
    '<text x="32" y="75" font-size="17" fill="#687587">Dense: purple · Sparse: magenta · Each point represents T/w/u, colored by temperature</text>'
  ];
  for (const [heading, leftFile, rightFile] of rows) {
    let height = 0;
    for (const [column, file, side] of [[0,leftFile,'Dense'], [1,rightFile,'Sparse']]) {
      const left = 32 + column * 784;
      texts.push('<text x="' + left + '" y="' + (y+26) + '" font-size="20" font-weight="600">' + escape(side + ' · ' + heading) + '</text>');
      const { data, info } = await sharp(path.join(assets,file)).resize({width:tileWidth})
        .flatten({background:'white'}).png().toBuffer({resolveWithObject:true});
      layers.push({input:data,left,top:y+42});
      height = Math.max(height,info.height);
    }
    y += height+84;
  }
  const svg = Buffer.from('<svg xmlns="http://www.w3.org/2000/svg" width="' + width + '" height="' + y + '"><g font-family="Arial, Helvetica, sans-serif" fill="#26364A">' + texts.join('') + '</g></svg>');
  await sharp({create:{width,height:y,channels:4,background:'white'}})
    .composite([...layers,{input:svg,left:0,top:0}]).png().toFile(path.join(assets,filename));
}

async function main() {
  const assets = path.resolve(process.argv[2] || path.join(__dirname,'assets'));
  const manifest = JSON.parse(fs.readFileSync(path.join(assets,'provenance.json'),'utf8'));
  for (const file of manifest.svg_files) {
    const output = path.join(assets,'preview',file.replace(/\.svg$/,'.png'));
    fs.mkdirSync(path.dirname(output),{recursive:true});
    await sharp(path.join(assets,file)).flatten({background:'white'}).png().toFile(output);
  }
  const pair = (id,style,title) => [title,
    'windows/dense_' + id + '_' + style + '.svg',
    'windows/sparse_' + id + '_' + style + '.svg'];
  await contactSheet(assets,'preview.png','Main Figure · Iteration 2',[
    pair('03_06_09','frames','Agents 3, 6, 9'),
    pair('01_03_05_07_09_11','frames','Six alternating agents'),
    pair('all_12','frames','All 12 agents'),
  ]);
  await contactSheet(assets,'all_windows_comparison.png','All 12 agents · Two window styles',[
    pair('all_12','frames','Frames and brackets'),
    pair('all_12','brackets','Brackets only'),
  ]);
  await contactSheet(assets,'three_agent_comparison.png','Three agents · Four selections',[
    pair('03_06_09','frames','Agents 3, 6, 9'),
    pair('05_06_07','frames','Agents 5, 6, 7'),
    pair('02_06_10','frames','Agents 2, 6, 10'),
    pair('01_06_12','frames','Agents 1, 6, 12'),
  ]);
  console.log('Rendered ' + manifest.svg_files.length + ' SVGs and three comparison sheets.');
}
main().catch(error => { console.error(error); process.exitCode=1; });
