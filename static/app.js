/**
 * Created by opopovyc on 5/3/17.
 */

var res;
var simulation_time;

$(document).ready(function() {
    openNav();
    $('#form').submit(function(){
        json_form  = $(this).serializeArray();
        simulation_time = parseFloat(json_form[36].value);
        $.post($(this).attr('action'), json_form, function(res){
            // Do something with the response `res`
            console.log(res);
            closeNav();
            setTimeout( plot, 500);
            // Don't forget to hide the loading indicator!
        });
    return false; // prevent default action
    });
});


/* Set the width of the side navigation to 250px and the left margin of the page content to 250px */
function openNav() {
    document.getElementById("mySidenav").style.width = "750px";
}

/* Set the width of the side navigation to 0 and the left margin of the page content to 0 */
function closeNav() {
    document.getElementById("mySidenav").style.width = "0";
}

function plot_temporal_layer_spikes(){
    $.get('/temporal_spikes', function(data){
        res = data;
        // console.log(data);

        $("<div>", {id: 'temporal_spike'}).appendTo('#chart_container').addClass('col-md-6');
        $("<div>", {id: 'temporal_spike_chart'}).appendTo('#temporal_spike');

        res = [[],[]];
        res[0].push("time");
        res[1].push("neuron");
        for (var i = 0; i < data['x'].length; i++){
            res[0].push(data['x'][i]);
            res[1].push(data['y'][i]);
        }

        var chart = c3.generate({
            bindto: '#temporal_spike_chart',
            title: {
              text: 'Temporal layer'
            },
            data: {
                xs:{
                    neuron: 'time',
                },
                columns: res,
                type: 'scatter'
            },
            axis: {
                x: {
                    tick: {
                        values: function () {
                            return d3.range(0, simulation_time + 1, 1);
                        }
                    }
                },
            },
            tooltip: {
                format: {
                    title: function (d) { return 'Time ' + d + "ms"; },
                    // value: function (value, ratio, id) { return value;}
                }
            },
            grid: {
                x: {show: true},
                y: {show: true}
            },
            subchart: {show: true}
        });
        chart.axis.range({min:{x:0.0}, max:{x: simulation_time}});
    });

};

function plot_som_layer_spikes(){
    $.get('/som_spike', function(data){
        res = data;
        // console.log(data);

        $("<div>", {id: 'som_spike'}).appendTo('#chart_container').addClass('col-md-6');
        $("<div>", {id: 'som_spike_chart'}).appendTo('#som_spike');

        res = [[],[]];
        res[0].push("time");
        res[1].push("neuron");
        for (var i = 0; i < data['x'].length; i++){
            res[0].push(data['x'][i]);
            res[1].push(data['y'][i]);
        }

        var chart = c3.generate({
            bindto: '#som_spike_chart',
            title: {
              text: 'SOM layer'
            },
            data: {
                xs:{
                    neuron: 'time',
                },
                columns: res,
                type: 'scatter'
            },
            axis: {
                x: {
                    tick: {
                        values: function () {
                            return d3.range(0, simulation_time + 1, 1);
                        }
                    }
                },
            },
            tooltip: {
                format: {
                    title: function (d) { return 'Time ' + d + "ms"; },
                    // value: function (value, ratio, id) { return value;}
                }
            },
            grid: {
                x: {show: true},
                y: {show: true}
            },
            subchart: {show: true}
        });
        chart.axis.range({min:{x:0.0}, max:{x: simulation_time}});
    });

};

function plot_inh_layer_spikes(){
    $.get('/inh_spike', function(data){
        res = data;
        // console.log(data);

        $("<div>", {id: 'inh_spike'}).appendTo('#chart_container').addClass('col-md-6');
        $("<div>", {id: 'inh_spike_chart'}).appendTo('#inh_spike');

        res = [[],[]];
        res[0].push("time");
        res[1].push("neuron");
        for (var i = 0; i < data['x'].length; i++){
            res[0].push(data['x'][i]);
            res[1].push(data['y'][i]);
        }

        var chart = c3.generate({
            bindto: '#inh_spike_chart',
            title: {
              text: 'Inh neuron spikes'
            },
            data: {
                xs:{
                    neuron: 'time',
                },
                columns: res,
                type: 'scatter'
            },
            axis: {
                x: {
                    tick: {
                        values: function () {
                            return d3.range(0, simulation_time + 1, 1);
                        }
                    }
                },
            },
            tooltip: {
                format: {
                    title: function (d) { return 'Time ' + d + "ms"; },
                    // value: function (value, ratio, id) { return value;}
                }
            },
            grid: {
                x: {show: true},
                y: {show: true}
            },
            subchart: {show: true}
        });
        chart.axis.range({min:{x:0.0, y: -0.5}, max:{x: simulation_time, y: 0.5}});
    });

};

function plot_temporal_layer_potential(){
    $.get('/membrane_potential_temporal_layer', function(data){
        // console.log(data);

        // $("<div>", {id: 'membrane_potential_temporal_layer'}).appendTo('#chart_container');
        $("<div>", {id: 'membrane_potential_temporal_layer_chart1'}).appendTo('#chart_container');
        $("<div>", {id: 'membrane_potential_temporal_layer_chart2'}).appendTo('#chart_container');
        $("<div>", {id: 'membrane_potential_temporal_layer_chart3'}).appendTo('#chart_container');

        var range1 = data['data'].slice(0,10)
        var range2 = data['data'].slice(10,20)
        range2.push(range1[0]);
        var range3 = data['data'].slice(20,30)
        range3.push(range1[0]);

        var chart1 = c3.generate({
            bindto: '#membrane_potential_temporal_layer_chart1',
            size: {
                height: 600
            },
            title: {
              text: 'Temporal layer membrane potential'
            },
            data: {
                x: 'time',
                columns: range1
            },
            axis: {
                x: {
                    tick: {
                        values: function () {
                            return d3.range(0, simulation_time + 1, 1);
                        }
                    }
                },
            },
            tooltip: {
                format: {
                    title: function (d) { return 'Time ' + d + "ms"; },
                    // value: function (value, ratio, id) { return value;}
                }
            },
            grid: {
                x: {show: true},
                y: {show: true}
            },
            subchart: {show: true}
        });
        chart1.axis.range({min:{x:0.0}, max:{x: simulation_time}});

        var chart2 = c3.generate({
            bindto: '#membrane_potential_temporal_layer_chart2',
            size: {
                height: 600
            },
            title: {
              text: 'Temporal layer membrane potential'
            },
            data: {
                x: 'time',
                columns: range2
            },
            axis: {
                x: {
                    tick: {
                        values: function () {
                            return d3.range(0, simulation_time + 1, 1);
                        }
                    }
                },
            },
            tooltip: {
                format: {
                    title: function (d) { return 'Time ' + d + "ms"; },
                    // value: function (value, ratio, id) { return value;}
                }
            },
            grid: {
                x: {show: true},
                y: {show: true}
            },
            subchart: {show: true}
        });
        chart2.axis.range({min:{x:0.0}, max:{x: simulation_time}});

        var chart3 = c3.generate({
            bindto: '#membrane_potential_temporal_layer_chart3',
            size: {
                height: 600
            },
            title: {
              text: 'Temporal layer membrane potential'
            },
            data: {
                x: 'time',
                columns: range3
            },
            axis: {
                x: {
                    tick: {
                        values: function () {
                            return d3.range(0, simulation_time + 1, 1);
                        }
                    }
                },
            },
            tooltip: {
                format: {
                    title: function (d) { return 'Time ' + d + "ms"; },
                    // value: function (value, ratio, id) { return value;}
                }
            },
            grid: {
                x: {show: true},
                y: {show: true}
            },
            subchart: {show: true}
        });
        chart3.axis.range({min:{x:0.0}, max:{x: simulation_time}});
    });

};

function plot(){
    plot_temporal_layer_spikes();
    plot_som_layer_spikes();
    plot_inh_layer_spikes();
    plot_temporal_layer_potential();
}


