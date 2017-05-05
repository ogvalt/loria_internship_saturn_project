/**
 * Created by opopovyc on 5/3/17.
 */

var res;

function get_data_for_plot(){
    $.get('/get_data_for_plot', function(data){
        res = data;
        console.log(data);
        res = [];
        for (var i = 0; i < data['x'].length; i++)
            res.push({ x: data['x'][i], y: data['y'][i]})

        var graph = new Rickshaw.Graph( {
	        element: document.querySelector("#chart"),
            width: 400,
            height: 500,
            renderer: 'scatterplot',
	        stroke: true,
            preserve: true,
	        series: [{
		        data: res,
		        color: 'steelblue',
                name: 'spike'
	        }]
            });
        graph.render();

        var hoverDetail = new Rickshaw.Graph.HoverDetail( {
            graph: graph,
            xFormatter: function(x) { return x + " ms" },
            yFormatter: function(y) { return y + " neuron" }
        } );

        var xAxis = new Rickshaw.Graph.Axis.X({
            graph: graph,
            ticksTreatment: 'glow'
        });
        xAxis.render();

        var yAxis = new Rickshaw.Graph.Axis.Y({
            graph: graph,
            ticksTreatment: 'glow'
        });
        yAxis.render();
    });
    // select .containter and append div with class attribute row
    // d3.select(".container").append("div").attr("class", "row")


};

function data_retrieval(){
    console.log(res);
};





