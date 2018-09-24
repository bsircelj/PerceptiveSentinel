const readline = require('readline');
const fs = require('fs');


var place = 'Grad_Jablje' // 'Grad_Jablje' or 'Aarhus' or 'Den_Helder'
const rl = readline.createInterface({
  input: fs.createReadStream('data\\data_' + place + '.json'),
  crlfDelay: Infinity
});

var array_of_data = [];

rl.on('line', (line) => {
  //console.log(`${line}`);
  array_of_data.push(JSON.parse(line).daily.data[0])
  //console.log(array_of_data)
  fs.writeFileSync("data\\data_" + place + "_array.json",JSON.stringify(array_of_data));
});