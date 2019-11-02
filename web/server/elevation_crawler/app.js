const csv = require('csv-parser');
const fs = require('fs');
const createCsvWriter = require('csv-writer').createObjectCsvWriter;

let csvFile = []

const csvWriter = createCsvWriter({
    path: 'DataSetWithElevation.csv',
    header: [
        { id: 'Amostra', title: 'Amostra' },
        { id: 'Latitude', title: 'Latitude' },
        { id: 'Longitude', title: 'Longitude' },
        { id: 'Elevation', title: 'Elevation' },
        { id: 'CA', title: 'Ca' },
        { id: 'MG', title: 'Mg' },
        { id: 'Na', title: 'Na' },
        { id: 'Cl', title: 'Cl' },
        { id: 'HCo3', title: 'HCo3' },
        { id: 'Co3', title: 'Co3' },
        { id: 'No3', title: 'No3' },
        { id: 'RS', title: 'RS' },
    ]
});


fs.createReadStream('mainDataSet1Original.csv')
    .pipe(csv())
    .on('data', (row) => {
        csvFile.push(row)
    })
    .on('end', () => {
        console.log(parseLocation(csvFile[1]))
        let location = parseLocation(csvFile[1])
        getAllLocation(csvFile, 1)
    });



function parseLocation(data) {
    return data.Latitude + ',' + data.Longitude
}



const https = require('https');



let rowWithErros = []

async function getAllLocation(csvData, index) {
    if (csvData.length > index) {
        location = parseLocation(csvData[index])

    } else {
        csvWriter
            .writeRecords(csvFile)
            .then(() => console.log('The CSV file was written successfully'));
        return
    }
    console
    https.get('https://api.jawg.io/elevations?locations=' + location + '&access-token=I0tZmMAxflhrsGdj0CfDBZEwFljBzlqcVQZz6cQPFfVecsCkmdGfi0odUi3HS7iD'
        , (resp) => {
            let data = '';
            // A chunk of data has been recieved.
            resp.on('data', (chunk) => {
                data += chunk;
                console.log(data)
            });

            // The whole response has been received. Print out the result.
            resp.on('end', async () => {
                let elevation = JSON.parse(data)[0].elevation
                csvFile[index].Elevation = elevation
                console.log("Ok!    Row" + index + " Elevation: " + elevation)
                await resolveAfterXMileSeconds(300)
                getAllLocation(csvData, index)
            });

        }).on("error", (err) => {
            console.err("Error!    Row: ", index)
            getAllLocation(csvData, index - 1)
            rowWithErros.push(index)
        });
}


function resolveAfterXMileSeconds(x) {
    return new Promise(resolve => {
        setTimeout(() => {
            resolve(x);
        }, x);
    });
}


console.log(rowWithErros)







function getLocation(location, index) {
    https.get('https://api.jawg.io/elevations?locations=' + location + '&access-token=I0tZmMAxflhrsGdj0CfDBZEwFljBzlqcVQZz6cQPFfVecsCkmdGfi0odUi3HS7iD'
        , (resp) => {
            let data = '';
            // A chunk of data has been recieved.
            resp.on('data', (chunk) => {
                data += chunk;
                console.log(data)
            });

            // The whole response has been received. Print out the result.
            resp.on('end', () => {
                let elevation = JSON.parse(data)[0].elevation
                console.log(elevation);
                csvFile[index].Elevation = elevation
                console.log(csvFile[index]);
                index++;
                csvWriter
                    .writeRecords([csvFile[index]])
                    .then(() => console.log('The CSV file was written successfully'));
            });

        }).on("error", (err) => {
            console.log("Error: " + err.message);
        });
}