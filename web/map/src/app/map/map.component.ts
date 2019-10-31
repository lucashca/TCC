import { Component, OnInit } from "@angular/core";
import { MapService } from "./map.service";
import { Coordinates } from "../class/coordinates";
import { Amostra } from "../class/amostra";
import { Nutriente } from "../class/nutriente";

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
  constructor(public mapService: MapService) { }

  amostras: Amostra[] = [];

  ngOnInit() {
    this.getDataSet();
    this.getNutrienteData(-12.7277777777778, -40.49722222222206)
    this.getNutrienteData(-12.9897222222222, -39.13027777777778)

  }

  placeMarker(event) {
    console.log(event);
    this.getNutrienteData(event.coords.lat, event.coords.lng);
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

      const cord = new Coordinates(lat, long);
      const n1 = new Nutriente("Ca", "mg/l", this.convertCa(parseFloat(a[3])));
      const n2 = new Nutriente("Mg", "mg/l", this.convertMg(parseFloat(a[4])));
      const n3 = new Nutriente("Na", "mg/l", this.convertNa(parseFloat(a[5])));
      const n4 = new Nutriente("K", "mg/l", this.convertK(parseFloat(a[6])));
      const nutrinetes: Nutriente[] = [n1, n2, n3, n4];
      const amostra = new Amostra(name, cord, nutrinetes);
      this.amostras.push(amostra);
    }
    console.log(this.amostras);
  }

  getNutrienteData(lat, lng) {
    this.mapService.getNutrienteData(lat, lng).subscribe(
      res => {
        const n1 = new Nutriente("Ca", "mg/l", this.convertCa(res.Ca));
        const n2 = new Nutriente("Mg", "mg/l", this.convertMg(res.Mg));
        const n3 = new Nutriente("Na", "mg/l", this.convertNa(res.Na));
        const n4 = new Nutriente("K", "mg/l", this.convertK(res.K));

        const nutrinetes: Nutriente[] = [n1, n2, n3, n4];
        const cord = new Coordinates(lat, lng);

        const amostra = new Amostra("Consulta", cord, nutrinetes);
        console.log(amostra)
        this.amostras.push(amostra);


      },
      error => { console.log(error) },
      () => { console.log("Finish") }
    );
  }
  // Lat -12.7277777777778
  // Lng -40.49722222222206

  /**[
          "PG-159",
         "-12.727777777777800",
         -40.497222222222206,
        87.8,
        60.0,
       26.5,
       "8.460000000000000",
       69.0
      ], */
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
