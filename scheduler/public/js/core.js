    //----------------------------------------------------------------------
        
    $(document).ready(function() 
    {
        
        //----------------------------------------------------------------------
    
        function limit_finished_jobs(limit)
        {

            if (limit < 0)
                $finished_table.ajax.url("/get_all_finished_jobs");
            else
                $finished_table.ajax.url("/get_all_finished_jobs?limit="+String(limit));
            
        }

        
        //----------------------------------------------------------------------
        //--------------------------- gen functions ----------------------------
        //----------------------------------------------------------------------
        
        function check_heartbeat_timestamp(aData)
        {
            var $red_bk_color = "#F78181"; //red
            var $bk_color = "#F78181"; //red
            //sample format
            //'2015-10-26 11:23:47.325000'
            if (aData.Status === "Offline")
            {
                return $bk_color;
            }
            else
            {
                var $date_arr = aData.Heartbeat.split(" ");
                if ($date_arr.length > 1)
                {
                    if (aData.Status === "Idle")
                        $bk_color = "#9FF781";  //green
                    else
                        $bk_color = "#F3F781";  // yellow
                    var $mDate = $date_arr[0].split("-");
                    if ($mDate.length < 3)
                        return $red_bk_color;
                    var $mTime = $date_arr[1].split(":");
                    if ($mTime.length < 3)
                        return $red_bk_color;
                    // Get the current date and check it to heartbeat
                    var $now = new Date();
                    if ($now.getFullYear() !== parseInt($mDate[0]))
                        return $red_bk_color;
                    if ($now.getMonth() + 1 !== parseInt($mDate[1]))
                        return $red_bk_color;
                    if ($now.getDate() !== parseInt($mDate[2]))
                        return $red_bk_color;
                    var $hourDiff = $now.getHours() - parseInt($mTime[0]);
                    var $minDiff = $now.getMinutes() - parseInt($mTime[1]);
                    if ($hourDiff > 1)
                        return $red_bk_color;
                    if ($minDiff > 10)
                        return $red_bk_color;
                }
            }
            return $bk_color;
        }
        
        //----------------------------------------------------------------------
        
        function convert_procmask_number_to_string(aData)
        {
            var $str = "";
        
            if(aData.Experiment === 'XRF')
            {
                if( (aData.ProcMask & 1) === 1)
                {
                    $str += '(A) Analyze datasets using ROI and ROI+ :: ';
                }
                if( (aData.ProcMask & 2) === 2)
                {
                    $str += '(B) Extract integrated spectra from analyzed files and fit the spectra to optimize fit parameters :: ';
                }
                if( (aData.ProcMask & 4) === 4)
                {
                    $str += '(C) Analyze datasets using ROI, ROI+ and per pixel fitting :: ';
                }
                if( (aData.ProcMask & 8) === 8)
                {
                    $str += '(D) :: ';
                }
                if( (aData.ProcMask & 16) === 16)
                {
                    $str += '(E) Add exchange information to analyzed files :: ';
                }
                if( (aData.ProcMask & 32) === 32)
                {
                    $str += '(F) Create hdf5 file from single line netcdf files for flyscan :: ';
                }
                if( (aData.ProcMask & 64) === 64)
                {
                    $str += '(G) Generate average .h5 from each detector :: ';
                }
            }
            else if (aData.Experiment === 'PTY')
            {
                $str = "PtychoLib";
            }
            return $str;
        }
        
        //----------------------------------------------------------------------
        
        function convert_status_number_to_string(aData)
        {
            var $str = "";
            if(aData.Status === 0)
            {
              $str = 'Waiting';
            }
            else if(aData.Status === 1)
            {
              $str = 'Processing';
            }
            else if(aData.Status === 2)
            {
              $str = 'Completed';
            }
            else if(aData.Status === 3)
            {
              $str = 'Canceling';
            }
            else if(aData.Status === 4)
            {
              $str = 'Canceled';
            }
            else if(aData.Status > 4)
            {
              $str = 'Error Occured';
            }
            return $str;
        }
        
        //----------------------------------------------------------------------

        function create_job_data_table($id, $url)
        {
            return $($id).DataTable(
            {
                //"processing": true,
                //"serverSide": true,
                "ajax": $url,
                "fnRowCallback": function( nRow, aData, iDisplayIndex, iDisplayIndexFull )
                {
                    var $status = convert_status_number_to_string(aData);
                    $('td', nRow).eq(3).html($status);
                    var $procmask = convert_procmask_number_to_string(aData);
                    $('td', nRow).eq(4).html($procmask);
                },
                "columns": [
                {
                    "class":          "details-control",
                    "orderable":      false,
                    "data":           null,
                    "defaultContent": ""
                },
                { "data": "Id" },
                { "data": "DataPath" },
                { "data": "Status" },
                { "data": "BeamLine" },
                {
                    "class":          "delete-control",
                    "orderable":      false,
                    "data":           null,
                    "defaultContent": ""
                }
                ],
                "order": [[1, 'desc']]
            });

        }
        
        //----------------------------------------------------------------------
        
        function find_node_id_by_hostname(hostname)
        {
            $node_list = $pn_table.data();
            $node_list.each( function (node)
            {
                if (node.Hostname === hostname)
                {
                    return d.Process_Node_Id;
                }
            });
            return -1;
        }
        
        //----------------------------------------------------------------------
        
        function format_job_details ( d ) 
        {
            var $host_and_port = '';
            var $host_name = '';
            $node_list = $pn_table.data();
            $node_list.each( function (node)
            {
                if (node.Id === d.Process_Node_Id)
                {
                    $host_and_port = node.Hostname+":"+node.Port;
                    $host_name = node.ComputerName;
                }
            });
            if(d.Experiment === 'XRF')
            {
            return  'Id: '+d.Id+'<br>'+
                    'DataPath: '+d.DataPath+'<br>'+
                    'Processing Type: '+convert_procmask_number_to_string(d)+'<br>'+
                    'Priority: '+d.Priority+'<br>'+
                    'Detector Elements: '+d.DetectorElements+'<br>'+
                    'Detector To Start With: '+d.DetectorToStartWith+'<br>'+
                    'Version: '+d.Version+'<br>'+
                    'Quick and Dirty: '+d.QuickAndDirty+'<br>'+
                    'NNLS: '+d.NNLS+'<br>'+
                    'XRF-Maps: '+d.XANES_Scan+'<br>'+
                    'BeamLine: '+d.BeamLine+'<br>'+
                    'Priority: '+d.Priority+'<br>'+
                    'Lines processed in parrallel: '+d.MaxLinesToProc+'<br>'+
                    'Files processed in parrallel: '+d.MaxFilesToProc+'<br>'+
                    'StartProcTime: '+d.StartProcTime+'<br>'+
                    'FinishProcTime: '+d.FinishProcTime+'<br>'+
                    'Dataset Files To Process: ' + d.DatasetFilesToProc +'<br>'+
                    'Process Node: '+$host_name+'<br>'+ 
                    'Email Addresses: '+d.Emails+'<br>'+ 
                    '<a href=/get_output_list?job_path='+d.DataPath+'>Output Directory</a> (output_old) <br>'+
                    '<a href=/get_output_list?job_path='+d.DataPath+'&process_type=PER_PIXEL>Output Directory</a> (output.fits) Per Pixel Fitting<br>'+
                    '<a href=http://'+$host_and_port+'/get_job_log?log_path='+d.Log_Path+'>Job Log</a>';
            }
            else if(d.Experiment === 'PTY')
            {
            return  'Id: '+d.Id+'<br>'+
                    'DataPath: '+d.DataPath+'<br>'+
                    'Priority: '+d.Priority+'<br>'+
                    'Version: '+d.Version+'<br>'+
                    'BeamLine: '+d.BeamLine+'<br>'+
                    'Priority: '+d.Priority+'<br>'+
                    'StartProcTime: '+d.StartProcTime+'<br>'+
                    'FinishProcTime: '+d.FinishProcTime+'<br>'+
                    'Dataset Files To Process: ' + d.DatasetFilesToProc +'<br>'+
                    'Process Node: '+$host_name+'<br>'+ 
                    'Email Addresses: '+d.Emails+'<br>'+ 
                    '<a href=http://'+$host_and_port+'/get_job_log?log_path='+d.Log_Path+'>Job Log</a>';
            }
            else
            {
                return 'Unknown type!';    
            }
        }

        //----------------------------------------------------------------------
        
        function format_pn_details ( d ) 
        {
            return  'Id: '+d.Id+'<br>'+
                    'Computer Name: '+d.ComputerName+'<br>'+
                    'Status: '+d.Status+'<br>'+
                    'NumThreads: '+d.NumThreads+'<br>'+
                    'Hostname: '+d.Hostname+'<br>'+
                    'Port: '+d.Port+'<br>'+
                    'Heartbeat: '+d.Heartbeat + '<br>'+
                    'Process Cpu Percent: '+d.ProcessCpuPercent + '%<br>'+
                    'Process Mem Percent: '+d.ProcessMemPercent + '%<br>'+
                    'System Cpu Percent: '+d.SystemCpuPercent + '%<br>'+
                    'System Mem Percent: '+d.SystemMemPercent + '%<br>'+
                    'System Swap Percent: '+d.SystemSwapPercent + '%';
        }
        
        //----------------------------------------------------------------------
        
        function gen_job_mask()
        {
            var procMask = 0;
            if ($('#analysis-type-a')[0].checked === true)
            {
                procMask += 1;
            }
            if ($('#analysis-type-b')[0].checked === true)
            {
                procMask += 2;
            }
            if ($('#analysis-type-c')[0].checked === true)
            {
                procMask += 4;
            }
            /*
            if ($('#analysis-type-d')[0].checked === true)
            {
                procMask += 8;
            }
            */
            if ($('#analysis-type-e')[0].checked === true)
            {
                procMask += 16;
            }
            /*
            if ($('#analysis-type-f')[0].checked === true)
            {
                procMask += 32;
            }
            */
            if ($('#analysis-type-g')[0].checked === true)
            {
                procMask += 64;
            }
            return procMask;
        }
        
        //----------------------------------------------------------------------

        function get_check_value(name)
        {
            if( $(name)[0].checked)
            {
                return 1;
            }
            return 0;
        }
        
        //----------------------------------------------------------------------
        
        function populate_datasets_list(str_datapath_id, str_dataset_list_id)
        {
            $.ajax(
            {
                type: 'POST',
                url:"/get_mda_list",
                data: {'job_path': $(str_datapath_id).val()},
                datatype: "json",
                success: function(data)
                {
                    //$.notify("Got dataset list", "success");
                    var $datasets = $.parseJSON(data);
                    $.notify("Got dataset list ", "success");
                    
                    $(str_dataset_list_id).html("");
                    $datasets.mda_files.sort();
                    for (i = 0; i < $datasets.mda_files.length; i++) 
                    { 
                        var node = $datasets.mda_files[i];
                        node = node.replace(/\\/g, "/");
                        var $shortname = node.split("/");
                        var $total = $shortname.length;
                        //var $option = "<option value=\""+$datasets.mda_files[i]+"\">"+$shortname[$total-1]+"</option>";
                        var $option = "<option value=\""+$shortname[$total-1]+"\">"+$shortname[$total-1]+"</option>";
                        $(str_dataset_list_id).append($option);
                    }
                },
                error: function(xhr, textStatus, errorThrown)
                {
                    $.notify(xhr.responseText,
                    {
                        autoHide: false,
                        clickToHide: true
                    });
                }
            });
        }
        
        function submit_new_xrf_job(is_xrf_maps_job)
        {
            var $procMask = gen_job_mask();
            var $quickNdirty = get_check_value('#option-quick-and-dirty');
            var $xrfbin = 0; // get_check_value('#option-xrf-bin');
            var $nnls = get_check_value('#option-nnls');
            var $xanes = is_xrf_maps_job; //0 = mapspy , 1 = xrf maps  get_check_value('#option-xanes');
            var $is_live_job = get_check_value('#option-is-live-job');
            var $dataset_filenames = 'all';
            var $pn_id = $( "#proc_node_option option:selected" ).attr('value');
            if ( $("#chk_all_datasets")[0].checked === false )
            {
                $dataset_filenames = $('#datasets_list').val().join();
            }
            
            if($procMask === 0 && $nnls === 0)
            {
                $.notify("No analysis type selected! Select A, B, C, D, E, or F", "error");
                return;
            }
            
            $.ajax(
            {
                type: 'POST',
                url:"/job",
                contentType: 'application/json; charset=utf-8',
                data: JSON.stringify({
                    'Experiment': 'XRF',
                    'BeamLine': '2-ID-E',
                    'DataPath': $("#DataPath").val(),
                    'Version': '9.00',
                    'Status': 0,
                    'Priority': parseInt($("#priority").val(), 10),
                    'StartProcTime': 0,
                    'FinishProcTime': 0,
                    'Log_Path': '',
                    'Emails': $("#option-emails").val(),
                    'Process_Node_Id': parseInt($pn_id, 10),
                    'ProcMask': $procMask,
                    'Standards': 'maps_standardinfo.txt', //$("#option-standard").val(),
                    'DetectorToStartWith': parseInt($("#option-detector-to-start-with").val(), 10),
                    'XRF_Bin': $xrfbin,
                    'MaxLinesToProc': parseInt($("#option-proc-per-line").val(), 10),
                    'MaxFilesToProc': parseInt($("#option-proc-per-file").val(), 10),
                    'DetectorElements': parseInt($("#option-detector-elements").val(), 10),
                    'XANES_Scan': $xanes,
                    'NNLS': $nnls,
                    'QuickAndDirty': $quickNdirty,
                    'Is_Live_Job': $is_live_job,
                    'DatasetFilesToProc': $dataset_filenames
                }),
                datatype: "json",
                success: function(data) 
                {
                    $.notify("Queued Job: " + $("#DataPath").val(), "success");
                    $queued_table.ajax.reload();
                },
                error: function(xhr, textStatus, errorThrown)
                {
                    $.notify(xhr.responseText,
                    {
                        autoHide: false,
                        clickToHide: true
                    });
                }
            });
        }
       
       function submit_new_pty_job()
        {
            
            var $calc_stxm = get_check_value('#pty-chk-calc-stxm');
            var $alg_epie = get_check_value('#pty-chk-alg-epie');
            var $alg_dm = get_check_value('#pty-chk-alg-dm');
            
            var $det_dist = $("#pty-detector-dist").val();
            var $pixel_size = $("#pty-dect-pix-size").val();
            
            var $center_y = $("#pty-diff-center-y").val();
            var $center_x = $("#pty-diff-center-x").val();
            var $diff_size = $("#pty-diff-size").val();
            var $rot = $( "#pty-rotate option:selected" ).attr('value');
            
            var $probe_size = $("#pty-probe-size").val();
            var $probe_modes = $("#pty-probe-modes").val();
            
            var $threshold = $("#pty-threshold").val();
            var $iter =  $("#pty-iterations").val();
            
            var $dataset_filenames = 'all';
            var $gpu_id = $( "#pty-gpu-node option:selected" ).attr('value');
            
            var $node_id = find_node_id_by_hostname('xrf1');
                       
            if ( $("#pty_chk_all_datasets")[0].checked === false )
            {
                $dataset_filenames = $('#pty_datasets_list').val().join();
            }
            
            $.ajax(
            {
                type: 'POST',
                url:"/job",
                contentType: 'application/json; charset=utf-8',
                data: JSON.stringify({
                    'Experiment': 'PTY',
                    'BeamLine': '2-ID-D',
                    'DataPath': $("#Pty-DataPath").val(),
                    'Version': '1.00',
                    'Status': 0,
                    'Priority': parseInt($("#pty-priority").val(), 10),
                    'StartProcTime': 0,
                    'FinishProcTime': 0,
                    'Log_Path': '',
                    'Emails': $("#pty-option-emails").val(),
                    'Process_Node_Id': $node_id,
                    'CalcSTXM': $calc_stxm,
                    'AlgorithmEPIE': $alg_epie,
                    'AlgorithmDM': $alg_dm,
                    'DetectorDistance': $det_dist,
                    'PixelSize': $pixel_size,
                    'CenterY': $center_y,
                    'CenterX': $center_x,
                    'DiffractionSize': $diff_size,
                    'Rotation': $rot,
                    'GPU_ID': $gpu_id,
                    'ProbeSize': $probe_size,
                    'ProbeModes': $probe_modes,
                    'Threshold': $threshold,
                    'Iterations': $iter,
                    'DatasetFilesToProc': $dataset_filenames
                }),
                datatype: "json",
                success: function(data) 
                {
                    $.notify("Queued Job: " + $("#Pty-DataPath").val(), "success");
                    $queued_table.ajax.reload();
                },
                error: function(xhr, textStatus, errorThrown)
                {
                    $.notify(xhr.responseText,
                    {
                        autoHide: false,
                        clickToHide: true
                    });
                }
            });
            
        }
       
        //----------------------------------------------------------------------
        //--------------------------- button clicks ----------------------------
        //----------------------------------------------------------------------        
        $("#Btn-Submit-Job").click(function(e) 
        {
            submit_new_xrf_job(0);
        });
        
        //----------------------------------------------------------------------
        
        $("#Btn-Submit-Xrf-Maps-Job").click(function(e) 
        {
            submit_new_xrf_job(1);
        });
        
        $("#Btn-Submit-Pty-Job").click(function(e) 
        {
            submit_new_pty_job();
        });
        
        //----------------------------------------------------------------------
        
        $('#Btn-Select-Dir').click(function(e)
        {
            $('.overlay').hide();
            var $tree = $('#jstree').jstree(true);
            var $sel = $tree.get_selected();
            $('#DataPath').val( $sel[0] );
            populate_datasets_list('#DataPath', '#datasets_list');
        });
        
        $('#Btn-Select-Pty-Dir').click(function(e)
        {
            $('.overlay').hide();
            var $tree = $('#pty-jstree').jstree(true);
            var $sel = $tree.get_selected();
            $('#Pty-DataPath').val( $sel[0] );
            populate_datasets_list('#Pty-DataPath', '#pty_datasets_list');
        });
        
        //----------------------------------------------------------------------
        
        $("#Btn-Browse-verify").click(function(e) 
        {
            $('.overlay').show();
            $('#jstree').jstree(
            {
                'core':
                {
                    'data' : function (obj, callback) 
                    {
                        var me = this;
                         $.ajax(
                        {
                            type: 'POST',
                            url:"/get_dataset_dirs_list",
                            data: {'job_path': 'verify', 'depth': 2},
                            datatype: "json",
                            success: function(rdata) 
                            {
                                callback.call(me, JSON.parse(rdata));
                            },
                            error: function(xhr, textStatus, errorThrown)
                            {
                                $.notify(xhr.responseText,  {
                                    autoHide: false,
                                    clickToHide: true
                                });
                            }
                        });
                    }
                },
                "plugins" : [ "sort" ]
            });
        });
        
         $("#Btn-Browse-production").click(function(e) 
        {
            $('.overlay').show();
            $('#jstree').jstree(
            {
                'core':
                {
                    'data' : function (obj, callback) 
                    {
                        var me = this;
                         $.ajax(
                        {
                            type: 'POST',
                            url:"/get_dataset_dirs_list",
                            data: {'job_path': 'production', 'depth': 2},
                            datatype: "json",
                            success: function(rdata) 
                            {
                                callback.call(me, JSON.parse(rdata));
                            },
                            error: function(xhr, textStatus, errorThrown)
                            {
                                $.notify(xhr.responseText,  {
                                    autoHide: false,
                                    clickToHide: true
                                });
                            }
                        });
                    }
                },
                 "plugins" : [ "sort" ]
            });
        });
        
        $("#Btn-Browse-cnm").click(function(e) 
        {
            $('.overlay').show();
            $('#jstree').jstree(
            {
                'core':
                {
                    'data' : function (obj, callback) 
                    {
                        var me = this;
                         $.ajax(
                        {
                            type: 'POST',
                            url:"/get_dataset_dirs_list",
                            data: {'job_path': 'cnm', 'depth': 1},
                            datatype: "json",
                            success: function(rdata) 
                            {
                                callback.call(me, JSON.parse(rdata));
                            },
                            error: function(xhr, textStatus, errorThrown)
                            {
                                $.notify(xhr.responseText,  {
                                    autoHide: false,
                                    clickToHide: true
                                });
                            }
                        });
                    }
                },
                "plugins" : [ "sort" ]
            });
        });

        $("#Btn-Browse-Pty").click(function(e) 
        {
            $('.overlay').show();
            $('#pty-jstree').jstree(
            {
                'core':
                {
                    'data' : function (obj, callback) 
                    {
                        var me = this;
                         $.ajax(
                        {
                            type: 'POST',
                            url:"/get_dataset_dirs_list",
                            data: {'job_path': 'pty', 'depth': 1},
                            datatype: "json",
                            success: function(rdata) 
                            {
                                callback.call(me, JSON.parse(rdata));
                            },
                            error: function(xhr, textStatus, errorThrown)
                            {
                                $.notify(xhr.responseText,  {
                                    autoHide: false,
                                    clickToHide: true
                                });
                            }
                        });
                    }
                },
                "plugins" : [ "sort" ]
            });
        });

        //----------------------------------------------------------------------
        
        jQuery('.tabs .tab-links a').on('click', function(e)  
        {
            var currentAttrValue = jQuery(this).attr('href');
            // Show/Hide Tabs
            jQuery('.tabs ' + currentAttrValue).show().siblings().hide();
            // Change/remove current tab to active
            jQuery(this).parent('li').addClass('active').siblings().removeClass('active');
            e.preventDefault();
        });
        
        //----------------------------------------------------------------------

        var $pn_table = $("#process_node_table").DataTable(
        {
            //"processing": true,
            //"serverSide": true,
            "ajax": "/process_node",
            "fnRowCallback": function( nRow, aData, iDisplayIndex, iDisplayIndexFull )
            {
                var $bk_color = check_heartbeat_timestamp(aData);
                $('td', nRow).css('background-color', $bk_color);
            },
            "columns": [
            {
                "class":          "details-control",
                "orderable":      false,
                "data":           null,
                "defaultContent": ""
            },
            { "data": "ComputerName" },
            { "data": "Status" },
            { "data": "NumThreads" },
            { "data": "Heartbeat" },
            { "data": "SystemCpuPercent" },
            { "data": "SystemMemPercent" },
            { "data": "SystemSwapPercent" }
            ],
            "order": [[1, 'asc']]
        });
       
       //-----------------------------------------------------------------------
       
       $("#chk_all_datasets").change(function()
       {
          if(this.checked)
          {
            $("#datasets_list").hide();   
          }
          else
          {
            $("#datasets_list").show();
          }
       });
       
        $("#pty_chk_all_datasets").change(function()
       {
          if(this.checked)
          {
            $("#pty_datasets_list").hide();   
          }
          else
          {
            $("#pty_datasets_list").show();
          }
       });
       
        //----------------------------------------------------------------------
        //--------------------------- init -------------------------------------
        //----------------------------------------------------------------------

        var $queued_table = create_job_data_table("#table_unprocessed_jobs", "/get_all_unprocessed_jobs");
        var $processing_table = create_job_data_table("#table_processing_jobs", "/get_all_processing_jobs");
        var $finished_table = create_job_data_table("#table_finished_jobs", "/get_all_finished_jobs?limit=100");
        $("#priority").val(5);
        $("#pty-priority").val(5);
        $("#pty-rotate").val(0);
        $("#chk_all_datasets").prop("checked", true);
        $("#pty_chk_all_datasets").prop("checked", true);
        
        $("#analysis-type-roi").prop("checked", true);
        $("#analysis-type-svd").prop("checked", true);
        
        $("#analysis-type-a").prop("checked", true);
        $("#analysis-type-g").prop("checked", true);
        $("#datasets_list").hide();
        $("#pty_datasets_list").hide();
        $('.overlay').hide();
        var $detailPnRows = [];
        var $detailJobRows = [];
        var $detailPRJobRows = [];
        var $detailQuJobRows = [];

        $("#limit_100").checked = true;
        
        $("#limit_100").on("click", function() { limit_finished_jobs(100); } );
        $("#limit_500").on("click", function() { limit_finished_jobs(500); } );
        $("#limit_1000").on("click", function() { limit_finished_jobs(1000); } );
        $("#limit_All").on("click", function() { limit_finished_jobs(-1); } );
        

        setInterval(function()
        {
            $queued_table.ajax.reload(null, false);
            $processing_table.ajax.reload(null, false);
        }, 4000);
	
        setInterval(function()
        {
            $pn_table.ajax.reload(null, false);
            $finished_table.ajax.reload(null, false);
        }, 10000);
        
        //----------------------------------------------------------------------
        
        //$('#table_finished_jobs').on( 'click', 'tr td.details-control', function () 
        $('.display_table').on( 'click', 'tr td.details-control', function () 
        {
            var tr = $(this).closest('tr');
            var detail_id = 0;
            var closest_table = tr.closest('table');
            var row = -1;
            var idx;
            if (closest_table[0].id === 'process_node_table')
            {
                row = $pn_table.row( tr );
                idx = $.inArray( tr.attr('id'), $detailPnRows );
                detail_id = 1;
            }
            else if(closest_table[0].id === 'table_unprocessed_jobs')
            {
                row = $queued_table.row( tr );
                idx = $.inArray( tr.attr('id'), $detailQuJobRows );
                detail_id = 2;
            }
            else if(closest_table[0].id === 'table_processing_jobs')
            {
                row = $processing_table.row( tr );
                idx = $.inArray( tr.attr('id'), $detailPRJobRows );
                detail_id = 3;
            }
            else if(closest_table[0].id === 'table_finished_jobs')
            {
                row = $finished_table.row( tr );
                idx = $.inArray( tr.attr('id'), $detailJobRows );
                detail_id = 4;
            }
            else
                return;
            //var row = $finished_table.row( tr );
            //var idx = $.inArray( tr.attr('id'), $detailJobRows );

            if ( row.child.isShown() ) 
            {
                tr.removeClass( 'details' );
                row.child.hide();

                // Remove from the 'open' array
                //$detailJobRows.splice( idx, 1 );
                if (detail_id === 1)
                    $detailPnRows.splice( idx, 1 )
                else if (detail_id === 2)
                    $detailQuJobRows.splice( idx, 1 )
                else if (detail_id === 3)
                    $detailPRJobRows.splice( idx, 1 )
                else if (detail_id === 4)
                    $detailJobRows.splice( idx, 1 )
            }
            else 
            {
                tr.addClass( 'details' );
                if (detail_id === 1)
                    row.child( format_pn_details( row.data() ) ).show();
                else
                    row.child( format_job_details( row.data() ) ).show();

                // Add to the 'open' array
                if ( idx === -1 ) 
                {
                    if (detail_id === 1)
                        $detailPnRows.push( tr.attr('id') );
                    else if (detail_id === 2)
                        $detailQuJobRows.push( tr.attr('id') );
                    else if (detail_id === 3)
                        $detailPRJobRows.push( tr.attr('id') );
                    else if (detail_id === 4)
                        $detailJobRows.push( tr.attr('id') );
                }
            }
        });
        
        $('.display_table').on( 'click', 'tr td.delete-control', function () 
        {
            var tr = $(this).closest('tr');
            var detail_id = 0;
            var closest_table = tr.closest('table');
            var row = -1;
            var idx = -1;
            if(closest_table[0].id === 'table_unprocessed_jobs')
            {
                row = $queued_table.row( tr );
                idx = $.inArray( tr.attr('id'), $detailQuJobRows );
                detail_id = 2;
            }
            else if(closest_table[0].id === 'table_processing_jobs')
            {
                row = $processing_table.row( tr );
                idx = $.inArray( tr.attr('id'), $detailPRJobRows );
                detail_id = 3;
            }
            else if(closest_table[0].id === 'table_finished_jobs')
            {
                row = $finished_table.row( tr );
                idx = $.inArray( tr.attr('id'), $detailJobRows );
                detail_id = 4;
            }
            else
                return;
            
            if (confirm("Are you sure you want to cancel this job?") == true) 
            {
                //send 
                $.ajax(
                {
                    type: 'DELETE',
                    url:"/job",
                    contentType: 'application/json; charset=utf-8',
                    data: JSON.stringify(row.data()),
                    datatype: "json",
                    success: function(data) 
                    {
                        $.notify("Canceling Job: " + row.data().Id, "success");
                    },
                    error: function(xhr, textStatus, errorThrown)
                    {
                        $.notify(xhr.responseText,
                        {
                            autoHide: false,
                            clickToHide: true
                        });
                    }
                });
                
            }
        });
        
        //----------------------------------------------------------------------
 
        $pn_table.on( 'draw', function () 
        {
            $.each( $detailPnRows, function ( i, id ) 
            {
                $('#'+id+' td.details-control').trigger( 'click' );
            });
            
            var $node_list1 = $pn_table.data();
            //$node_list1.sort();
            $node_list1.each( function (node)
            {
                if ( $("#proc_node_option option[value='"+node.Id+"']").length < 1 )
                {
                    $('#proc_node_option').append( $("<option></option>").attr('value',node.Id).text(node.ComputerName));
                }
            });
            
        });
 
         //----------------------------------------------------------------------

        $queued_table.on( 'draw', function () 
        {
            $.each( $detailQuJobRows, function ( i, id ) 
            {
                $('#'+id+' td.details-control').trigger( 'click' );
            });
        });

         //----------------------------------------------------------------------
 
        $processing_table.on( 'draw', function () 
        {
            $.each( $detailPRJobRows, function ( i, id ) 
            {
                $('#'+id+' td.details-control').trigger( 'click' );
            });
        });
 
         //----------------------------------------------------------------------
 
        $finished_table.on( 'draw', function () 
        {
            $.each( $detailJobRows, function ( i, id ) 
            {
                $('#'+id+' td.details-control').trigger( 'click' );
            });
        });
        
        //----------------------------------------------------------------------
        
    });