import { chromium } from 'playwright';
import path from 'node:path';
import os from 'node:os';

(async () => {
  const browser = await chromium.launch({ channel: 'chrome', headless: true, args: ['--enable-webgl', '--ignore-gpu-blocklist'] });
  const page = await browser.newPage({ viewport: { width: 1672, height: 941 } });
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  page.on('console', message => { if(message.type() === 'error') errors.push(message.text()); });
  page.on('response', response => { if(response.status() >= 400 && response.url().startsWith('http://localhost:3000')) errors.push(`${response.status()} ${response.url()}`); });
  await page.goto('http://localhost:3000/login');
  await page.locator('input[name=email]').fill('dr.sharma@onco.ai');
  await page.locator('input[name=password]').fill('demo');
  await page.getByRole('button', { name:'Sign in', exact:true }).click();
  await page.waitForURL('**/dashboard');
  await page.locator('.digital-twin-card canvas').waitFor();
  await page.waitForTimeout(3500);
  for (const [width,height] of [[1672,941],[1920,1080],[1440,900],[1366,768],[1024,768]]) {
    await page.setViewportSize({width,height});
    await page.waitForTimeout(500);
    const screenshot = path.join(os.tmpdir(), `onco-reference-${width}.png`);
    await page.screenshot({path:screenshot,fullPage:true});
    const dimensions = await page.evaluate(() => ({ width:innerWidth, content:document.documentElement.scrollWidth, canvases:document.querySelectorAll('canvas').length }));
    if (dimensions.content > width + 1) errors.push(`Horizontal overflow at ${width}: ${dimensions.content}`);
    console.log(JSON.stringify({ viewport:[width,height], screenshot, ...dimensions }));
  }
  await page.setViewportSize({width:1440,height:900});
  for (const mode of ['CT Scan','Pathology','Gene Map','3D View']) {
    await page.getByRole('tab',{name:mode,exact:true}).click();
    if (await page.getByRole('tab',{name:mode,exact:true}).getAttribute('aria-selected') !== 'true') errors.push(`Mode ${mode} not selected`);
  }
  await page.getByRole('button',{name:'Reset digital twin view'}).click();
  await page.getByRole('button',{name:'Mute voice'}).click();
  const downloadEvent = page.waitForEvent('download');
  await page.getByRole('button',{name:'Export PDF Brief'}).click();
  const download = await downloadEvent;
  if (!(download.suggestedFilename().endsWith('.pdf'))) errors.push('PDF export did not produce a PDF');
  await page.getByRole('button',{name:/3 clinical trials match/}).click();
  await page.locator('.modal-backdrop').waitFor();
  await page.locator('.modal-close').click();
  await page.getByRole('button',{name:/Run AI Analysis/}).click();
  await page.locator('.processing-overlay').waitFor({state:'visible'});
  await page.locator('.processing-overlay').waitFor({state:'hidden'});
  const result = await page.request.post('http://localhost:3000/api/slm/chat', {data:{patientId:'ONC-2048',inputVersion:'qa',message:'Summarize this patient'}});
  if (!result.ok()) errors.push(`Authenticated copilot: ${result.status()}`);
  const routes = ['/overview','/patients','/patients/ONC-2048?tab=Labs','/ml','/dl','/nlp','/slm','/safety-lab','/tumor-board/ONC-2048','/integrated-analysis/ONC-2048','/analytics','/audit','/settings'];
  let tabCount = 0;
  for (const route of routes) {
    const response = await page.goto('http://localhost:3000'+route);
    await page.locator('.page-wrap').waitFor();
    if (response.status() !== 200) errors.push(`${route}: ${response.status()}`);
    const tabs = page.locator('.tabs button');
    for (let i=0;i<await tabs.count();i++) { await tabs.nth(i).click(); tabCount++; await page.waitForTimeout(80); }
    console.log(`PASS route and tabs: ${route}`);
  }
  await page.goto('http://localhost:3000/patients?create=true');
  await page.locator('.modal-backdrop').waitFor();
  await page.goto('http://localhost:3000/nlp');
  await page.locator('input[type=file]').setInputFiles({name:'synthetic-qa.txt',mimeType:'text/plain',buffer:Buffer.from('Synthetic QA report. No real patient information.')});
  await page.getByRole('status').waitFor();
  await page.goto('http://localhost:3000/ml');
  await page.getByRole('button',{name:/Run prediction/}).click();
  await page.locator('.success-line').filter({hasText:'Prediction complete'}).waitFor();
  await page.goto('http://localhost:3000/overview');
  await page.setViewportSize({width:1024,height:768});
  await page.getByRole('button',{name:'Clinical Copilot',exact:true}).click();
  if (!(await page.locator('.copilot-dock').isVisible())) errors.push('Tablet copilot drawer not visible');
  await page.getByRole('button',{name:'Close copilot'}).click();
  console.log(JSON.stringify({tabCount,errors}));
  await browser.close();
  if (errors.length) process.exitCode=1;
})().catch(error => { console.error(error); process.exitCode=1; });
