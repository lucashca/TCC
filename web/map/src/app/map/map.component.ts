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
  center = new Coordinates(-12.53536283971304, -39.834527876275786);
  zoom = 9;

  markers: Coordinates[] = [];
  userMarkers: Coordinates[] = [];
  constructor(public mapService: MapService) {}

  amostras: Amostra[] = [];

  ngOnInit() {
    this.getDataSet();
  }

  placeMarker(event) {
    console.log(event);
    this.userMarkers.push(new Coordinates(event.coords.lat, event.coords.lng));
  }

  getDataSet() {
    this.mapService.getMarkers().subscribe(res => {
      this.parseAmostra(res);
    });
  }

  parseAmostra(data) {
    this.amostras = [];
    for (let a of data) {
      const lat = parseFloat(a[0].replace(",", "."));
      const long = parseFloat(a[1].replace(",", "."));

      const cord = new Coordinates(lat, long);
      const n1 = new Nutriente("Ca", "", parseFloat(a[2]));
      const n2 = new Nutriente("Mg", "", parseFloat(a[3]));
      const n3 = new Nutriente("Na", "", parseFloat(a[4]));
      const n4 = new Nutriente("K", "", parseFloat(a[5]));
      const n5 = new Nutriente("Cl", "", parseFloat(a[6]));

      const nutrinetes: Nutriente[] = [n1, n2, n3, n4, n5];
      const amostra = new Amostra(cord, nutrinetes);
      this.amostras.push(amostra);
    }
    console.log(this.amostras);
  }
}
