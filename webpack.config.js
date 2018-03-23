var webpack = require('webpack');
var path = require('path');

module.exports = {
    mode: 'production',
    devtool: 'source-map',
    entry: './src/index.ts',
    output: {
        path: path.resolve(__dirname, 'dist'),
        filename: 'index.js',
        publicPath: '/dist/',
        libraryTarget: 'umd'
    },
    resolve: {
        extensions: ['.ts', 'tsx', '.js']
    },
    module: {
        rules: [
            {
                test: /\.tsx?$/,
                use: 'ts-loader',
                exclude: /node_modules/
            }
        ]
    },

    plugins: [
        new webpack.optimize.ModuleConcatenationPlugin()
    ]

};