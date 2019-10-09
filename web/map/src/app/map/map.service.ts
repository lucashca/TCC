import { Injectable } from "@angular/core";

import { Coordinates } from "../class/coordinates";

import { cords } from "./markers";

import "./markers.js";
import { HttpClient } from "@angular/common/http";

@Injectable({
  providedIn: "root"
})
export class MapService {
  markers: Coordinates[] = [];

  constructor(public http: HttpClient) {}

  setMarkers() {
    for (let c of cords) {
      this.markers.push(new Coordinates(c[0], c[1]));
    }
  }

  getMarkers() {
    this.setMarkers();

    return this.http.get("assets/dataset.json");
  }
}
