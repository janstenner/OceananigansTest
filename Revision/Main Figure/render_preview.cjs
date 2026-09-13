// Optional PNG contact sheet and visual QA. SVG generation only needs Julia.
// Requires Node.js and sharp (available in the Codex bundled runtime).
const fs = require('node:fs');
const path = require('node:path');
const sharp = require('sharp');

async function main() {
  const assets = path.resolve(process.argv[2] || path.join(__dirname, 'assets'));
  const qa = path.join(assets, 'preview');
  fs.mkdirSync(qa, { recursive: true });
  const files = fs.readdirSync(assets).filter(name => name.endsWith('.svg'));
  for (const name of files) {
    await sharp(path.join(assets, name)).flatten({ background: 'white' })
      .png().toFile(path.join(qa, name.replace('.svg', '.png')));
  }
  const width = 1600, tileWidth = 754;
  const rows = [
    ['Temperature at all 384 probe locations', 'GO-GC: 12 retained probe locations',
      'dense_temperature.svg', 'sparse_temperature.svg'],
    ['Dense observations with agent windows', 'Sparse observations with agent windows',
      'dense_temperature_windows.svg', 'sparse_temperature_windows.svg'],
    ['Three local dense windows', 'Three local sparse windows',
      'dense_local_windows.svg', 'sparse_local_windows.svg'],
  ];
  let y = 100;
  const layers = [];
  const texts = ['<text x="32" y="43" font-size="30" font-weight="600">Main Figure · Dense / Sparse</text>',
    '<text x="32" y="75" font-size="17" fill="#687587">Same two-plume state · frozen Varying-IC GO-GC mask · true 15-column windows</text>'];
  for (const [leftTitle, rightTitle, leftFile, rightFile] of rows) {
    let height = 0;
    for (const [column, file, title] of [[0, leftFile, leftTitle], [1, rightFile, rightTitle]]) {
      const left = 32 + column * 784;
      texts.push(`<text x="${left}" y="${y + 26}" font-size="20" font-weight="600">${title}</text>`);
      const { data, info } = await sharp(path.join(assets, file)).resize({ width: tileWidth })
        .flatten({ background: 'white' }).png().toBuffer({ resolveWithObject: true });
      layers.push({ input: data, left, top: y + 42 });
      height = Math.max(height, info.height);
    }
    y += height + 84;
  }
  const textSvg = Buffer.from(`<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${y}"><g font-family="Arial, Helvetica, sans-serif" fill="#26364A">${texts.join('')}</g></svg>`);
  await sharp({ create: { width, height: y, channels: 4, background: 'white' } })
    .composite([...layers, { input: textSvg, left: 0, top: 0 }]).png()
    .toFile(path.join(assets, 'preview.png'));
  console.log(`Rendered ${files.length} SVGs and ${path.join(assets, 'preview.png')}`);
}
main().catch(error => { console.error(error); process.exitCode = 1; });
