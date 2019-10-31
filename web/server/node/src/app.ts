import express from 'express'
import cors from 'cors'


class App {
    public express: express.Application;

    public constructor() {
        this.express = express()
        this.middlewares()
        this.routes()
    }

    private middlewares(): void {
        this.express.use(express.json())
        this.express.use(cors())
    }

    private routes(): void {
        this.routeHelloWorld()
        this.routeGetNutrientesData()
    }

    private routeHelloWorld(): void {
        this.express.get('/', (_req, res) => {

            return res.send('Hello World')
        })
    }

    private routeGetNutrientesData(): void {
        this.express.post('/getNutrientesData', (req, res) => {
            let body = req.body
            console.log(body)
            res.status(200).end()
        })
    }
}

export default new App().express
