import { Component, OnInit } from "@angular/core";
import { MapService } from "./map.service";
import { Coordinates } from "../class/coordinates";
import { Amostra } from "../class/amostra";
import { Nutriente } from "../class/nutriente";
import { cords } from './markers';
import { ControlPosition } from '@agm/core/services/google-maps-types';

@Component({
  selector: "app-map",
  templateUrl: "./map.component.html",
  styleUrls: ["./map.component.css"]
})
export class MapComponent implements OnInit {
  userIcon = "../../assets/images/icons/marker-blue-40.png";
  defaultIcon = "../../assets/images/icons/marker-red-40.png";
  center = new Coordinates(-13.19503, -39.973854);
  zoom = 9;

  markers: Coordinates[] = [];
  userMarkers: Coordinates[] = [];




  test: Amostra[]


  constructor(public mapService: MapService) { }

  amostras: Amostra[] = [];

  ngOnInit() {
    this.getDataSet();

  }

  getElevation(lat, lng) {
    let location = lat + ',' + lng
    this.mapService.getElevation(location).subscribe(
      res => { console.log(res); },
      err => {
        console.error(err);
      }
    )

  }


  getKNeighbors(k, cord) {

    let neighbors = []
    for (let a of this.amostras) {
      const cord2 = a.cord
      const dist = this.calulateLatLngDistance(cord, cord2)

      if (neighbors.length < k) {
        neighbors.push({ amostra: a, dist: dist })
      } else {
        for (let index in neighbors) {
          const d = neighbors[index].dist
          if (dist < d) {
            neighbors[index] = { amostra: a, dist: dist }
            break
          }
        }
      }
    }
    return this.createAmostra(neighbors, cord)

  }

  sortArrayByLess(neighbors) {
    let cont = 0;
    let notSorted = true;
    while (notSorted) {
      notSorted = false
      for (let n of neighbors) {
        if (cont + 1 < neighbors.length) {
          let n2 = neighbors[cont + 1]
          if (n2.dist < n.dist) {
            neighbors[cont] = n2
            neighbors[cont + 1] = n
            notSorted = true
          }
        }
        cont++;
      }
    }
    return neighbors
  }

  getWeights(neighbors) {
    let weights = []
    for (let n of neighbors) {
      weights.push(1 / n.dist)
    }
    let total = this.sumArray(weights)

    for (let i in weights) {
      weights[i] = weights[i] / total
    }
    return weights;
  }

  createAmostra(neighbors: any[], cord: Coordinates) {
    neighbors = this.sortArrayByLess(neighbors)

    let weights = this.getWeights(neighbors)

    let nutre = this.createArrayWithValToArray(0, neighbors[0].amostra.nutrientes.length)

    let w = 0;
    for (let n of neighbors) {
      let cont = 0;
      for (let nut of n.amostra.nutrientes) {
        nutre[cont] += nut.value * weights[w]
        cont++

      }
      w++;
    }
    const n1 = new Nutriente("Ca", "mg/l", nutre[0]);
    const n2 = new Nutriente("Mg", "mg/l", nutre[1]);
    const n3 = new Nutriente("Na", "mg/l", nutre[2]);
    const n4 = new Nutriente("K", "mg/l", nutre[3]);
    const nutrinetes: Nutriente[] = [n1, n2, n3, n4];
    let amostra = new Amostra("Predita", cord, 0, nutrinetes)
    return amostra
    //this.amostras.push(amostra)

  }
  createArrayWithValToArray(val, length) {
    let ar = []
    for (let i = 0; i < length; i++) {
      ar[i] = val
    }

    return ar
  }

  sumArray(arr) {
    let total = 0
    for (let d of arr) {
      total += d;
    }
    return total;
  }

  calulateLatLngDistance(cord: Coordinates, cord2: Coordinates) {
    // Raio da terra = 6371km
    let rt = 6371 * 1000

    let lat1 = cord.latitude * Math.PI / 180
    let lat2 = cord2.latitude * Math.PI / 180
    let lng1 = cord.longitude * Math.PI / 180
    let lng2 = cord2.longitude * Math.PI / 180

    let lat = lat2 - lat1
    let lng = lng2 - lng1

    let val1 = Math.pow(Math.sin(lat / 2), 2) + Math.cos(lat1) * Math.cos(lat2) * Math.pow(Math.sin(lng / 2), 2)
    let val2 = Math.atan2(Math.sqrt(val1), Math.sqrt(1 - val1))
    let val3 = rt * val2
    return val3
  }
  calculateEuclidianDistance(cord: Coordinates, cord2: Coordinates) {

    return Math.sqrt(Math.pow(cord2.latitude - cord.latitude, 2) + Math.pow(cord2.longitude - cord.longitude, 2))

  }

  cont = 0;

  placeMarker(event) {

    this.cont++;
    let amostra = this.getKNeighbors(5, new Coordinates(event.coords.lat, event.coords.lng))
    this.amostras.push(amostra)
    this.metricas()

  }

  randomInt(n) {
    return Math.round(Math.random() * n)
  }
  cloneDataset(dataSet) {
    let clD = dataSet.map((a) => a)
    return clD
  }
  cloneNutriente(n: Nutriente) {
    return new Nutriente(n.name, n.unidade, n.value)
  }
  cloneAmostra(amostra: Amostra) {
    let nut: Nutriente[] = []
    for (let n of amostra.nutrientes) {
      nut.push(this.cloneNutriente(n))
    }
    return new Amostra(amostra.name, amostra.cord, amostra.elevation, nut)
  }

  trainTestSlplit(dataSet: Amostra[], testSize: number) {
    let treino = []
    let test = []
    let clonedDataset = this.cloneDataset(dataSet)
    let testLength = Math.round(dataSet.length * testSize)
    let trainLength = dataSet.length - testLength
    for (let i = 0; i < testLength; i++) {
      let index = this.randomInt(dataSet.length)
      if (clonedDataset[index] != null) {
        test.push(this.cloneAmostra(clonedDataset[index]))
        clonedDataset[index] = null
      }
    }
    for (let d of clonedDataset) {
      if (d != null) {
        //console.log(d)
        treino.push(d)
      }
    }
    return [treino, test]




  }



  getDataSet() {
    this.mapService.getMarkers().subscribe(res => {
      this.parseAmostra(res);
    });
  }

  parseAmostra(data) {
    this.amostras = [];
    for (let a of data) {
      const name = a[0];
      const lat = parseFloat(a[1]);
      const long = parseFloat(a[2]);
      const elevation = parseFloat(a[3])
      const cord = new Coordinates(lat, long);
      const n1 = new Nutriente("Ca", "mg/l", this.convertCa(parseFloat(a[4])));
      const n2 = new Nutriente("Mg", "mg/l", this.convertMg(parseFloat(a[5])));
      const n3 = new Nutriente("Na", "mg/l", this.convertNa(parseFloat(a[6])));
      const n4 = new Nutriente("K", "mg/l", this.convertK(parseFloat(a[7])));
      const nutrinetes: Nutriente[] = [n1, n2, n3, n4];
      const amostra = new Amostra(name, cord, elevation, nutrinetes);
      this.amostras.push(amostra);
    }
    let res = this.trainTestSlplit(this.amostras, 0.3)
    this.amostras = res[0]
    this.test = res[1]
  }

  predictDatas(amostras: Amostra[], k) {
    let pred: Amostra[] = []
    for (let a of amostras) {
      let p = this.getKNeighbors(k, a.cord)
      pred.push(p)
    }

    return pred;
  }

  getNutrienteDataByColumn(amostra: Amostra[], column: number) {
    let nut = []
    for (let p of amostra) {
      nut.push(p.nutrientes[column].value)
    }

    return nut;
  }

  getYTrueYPred(amostraTrue: Amostra[], amostraPred: Amostra[], column: number) {
    let yTrue = this.getNutrienteDataByColumn(amostraTrue, column)
    let yPred = this.getNutrienteDataByColumn(amostraPred, column)
    return [yTrue, yPred]
  }
  metricas() {
    let resultados = []
    for (let k = 1; k < 100; k++) {
      console.log("Predict K:", k)

      let pred = this.predictDatas(this.test, k)
      let res = []
      for (let i = 0; i < 4; i++) {
        let ytyp = this.getYTrueYPred(this.test, pred, i)
        let mse = this.mean_squared_error(ytyp[0], ytyp[1])
        res.push(mse);
      }
      console.log("MSE")
      console.log("Ca Mg Na K")
      console.log(res)

    }
    console.log(resultados)
  }

  mean_squared_error(y_true, y_pred) {
    let mse = 0
    for (let i = 0; i < y_true.length; i++) {
      const yt = y_true[i];
      const yp = y_pred[i]
      mse += Math.pow(yt - yp, 2)
    }
    return mse / y_pred.length
  }



























  getNutrienteData(lat, lng) {
    this.mapService.getNutrienteData(lat, lng).subscribe(
      r => {
        let res: any = r;
        const n1 = new Nutriente("Ca", "mg/l", this.convertCa(res.Ca));
        const n2 = new Nutriente("Mg", "mg/l", this.convertMg(res.Mg));
        const n3 = new Nutriente("Na", "mg/l", this.convertNa(res.Na));
        const n4 = new Nutriente("K", "mg/l", this.convertK(res.K));

        const nutrinetes: Nutriente[] = [n1, n2, n3, n4];
        const cord = new Coordinates(lat, lng);
        const elevation = 0;
        const amostra = new Amostra("Consulta", cord, elevation, nutrinetes);

        this.amostras.push(amostra);


      },
      error => { console.log(error) },
      () => { console.log("Finish") }
    );
  }

  convertCa(val) {
    return val / 200.4
  }
  convertMg(val) {
    return val / 121.56

  }
  convertNa(val) {
    return val / 230

  }
  convertK(val) {
    return val / 391

  }

}
